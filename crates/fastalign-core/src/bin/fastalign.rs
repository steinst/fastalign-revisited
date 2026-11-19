//! `fastalign`: native command line for training fast_align's model, saving it, forced
//! alignment with a saved model (or with a fast_align `-p` table) and symmetrisation.
//!
//! fast_align's short flags are accepted where they do not clash with the sister tool
//! `gizapp` (`-o`, `-s`, `-t`, `-q` keep gizapp's meaning): see `fastalign train --help`.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use fastalign_core::corpus::{Corpus, LoadOptions};
use fastalign_core::io::{Direction, Model, SubwordInfo};
use fastalign_core::oracle::{force_align, ForceOptions, ForceResult, TextTable, TrainOptions};
use fastalign_core::pipeline::{train, with_threads, Directions, Progress, TrainConfig};
use fastalign_core::subword::Boundary;
use fastalign_core::symmetrize::{symmetrize_text, Method};

#[derive(Parser)]
#[command(
    name = "fastalign",
    version,
    about = "fast_align's reparameterised IBM Model 2, byte-identical to fast_align"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train on a parallel corpus and write alignments.
    Train(TrainArgs),
    /// Align new sentence pairs with a saved model, or with a fast_align `-p` table.
    Align(AlignArgs),
    /// Symmetrise two directional alignment files (as `atools -c METHOD`).
    Symmetrize(SymArgs),
    /// Write a saved model's table in fast_align's `-p` text format.
    ExportTable(ExportArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Dirs {
    Both,
    Fwd,
    Inv,
}

impl Dirs {
    fn directions(self) -> Directions {
        match self {
            Dirs::Both => Directions::Both,
            Dirs::Fwd => Directions::Forward,
            Dirs::Inv => Directions::Inverse,
        }
    }
}

/// Word boundaries of an already segmented corpus. The native binary does not train or run
/// SentencePiece (or any tokeniser): segment the text first (the Python package does it with
/// `fastalign train --subword`), then say here how the pieces mark word boundaries.
#[derive(Args, Clone)]
struct SubwordArgs {
    /// The input is already cut into subword pieces; a piece starting with MARKER (SentencePiece:
    /// `▁`) begins a word. Alignments are projected to words in every output file.
    #[arg(long, value_name = "MARKER", conflicts_with = "subword_continuation")]
    subword_marker: Option<String>,
    /// As --subword-marker for subword-nmt style text: a piece ending with MARKER (`@@`)
    /// continues into the next piece.
    #[arg(long, value_name = "MARKER")]
    subword_continuation: Option<String>,
}

impl SubwordArgs {
    fn boundary(&self) -> R<Option<Boundary>> {
        let (kind, m) = match (&self.subword_marker, &self.subword_continuation) {
            (Some(m), None) => ("start", m),
            (None, Some(m)) => ("continue", m),
            _ => return Ok(None),
        };
        Ok(Some(Boundary::from_parts(kind, m.as_bytes())?))
    }
}

/// Model options shared by `train` and `align --table`.
#[derive(Args, Clone)]
struct ModelArgs {
    /// Do not favour the diagonal (fast_align without `-d`).
    #[arg(long)]
    no_diagonal: bool,
    /// Starting tension of the diagonal prior (fast_align `-T`).
    #[arg(short = 'T', long, default_value_t = 4.0)]
    tension: f64,
    /// Probability of the NULL word with the diagonal prior (fast_align `-q`).
    #[arg(long, default_value_t = 0.08)]
    p0: f64,
    /// No NULL word (fast_align `-N`).
    #[arg(short = 'N', long)]
    no_null: bool,
}

#[derive(Args)]
struct TrainArgs {
    /// Corpus: one file of `source ||| target` lines, or the source text (then give TGT).
    input: PathBuf,
    /// Target text, line-aligned with INPUT.
    tgt: Option<PathBuf>,
    /// Output prefix: writes PREFIX.fwd.pharaoh, PREFIX.inv.pharaoh and
    /// PREFIX.<method>.pharaoh for every --symmetrize. Without it the single direction chosen
    /// by --directions is written to standard output.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    #[arg(long, value_enum, default_value = "both")]
    directions: Dirs,
    /// Iterations (fast_align `-I`).
    #[arg(short = 'I', long, default_value_t = 5)]
    iterations: i32,
    #[command(flatten)]
    model: ModelArgs,
    /// Do not optimise the tension (fast_align without `-o`).
    #[arg(long)]
    no_optimize_tension: bool,
    /// Plain EM instead of variational Bayes (fast_align without `-v`).
    #[arg(long)]
    no_variational_bayes: bool,
    /// Dirichlet prior of variational Bayes (fast_align `-a`).
    #[arg(short = 'a', long, default_value_t = 0.01)]
    alpha: f64,
    /// Lines per statistics buffer (fast_align `-b`); changes only how sums are grouped,
    /// as in fast_align.
    #[arg(short = 'b', long, default_value_t = 10000)]
    buffer_size: u64,
    /// Skip lines with more tokens than this on either side (default: keep all; dropping
    /// lines changes the result relative to fast_align on the unfiltered file).
    #[arg(long)]
    max_len: Option<usize>,
    /// Symmetrisation heuristics to write (repeatable). Needs both directions and --output.
    #[arg(short = 's', long)]
    symmetrize: Vec<Method>,
    /// Append ` ||| score` (log probability of the best alignment) to the directional lines.
    #[arg(long)]
    scores: bool,
    /// Do not build the dense lookup index of the busiest table rows (saves 64 MB per
    /// direction, slower E-step and M-step; no output changes).
    #[arg(long)]
    no_lookup_index: bool,
    #[command(flatten)]
    subword: SubwordArgs,
    /// With --subword-marker/--subword-continuation: also write the piece-level alignments of
    /// each direction (PREFIX.<dir>.subword.pharaoh); positions are pieces, not words.
    #[arg(long, requires = "output")]
    subword_output: bool,
    /// Save the trained model here.
    #[arg(long)]
    save: Option<PathBuf>,
    /// Also write fast_align's `-p` conditional probability tables (PREFIX.<dir>.table).
    #[arg(long, requires = "output")]
    table: bool,
    /// Beam threshold of the tables (fast_align `-t`).
    #[arg(long, default_value_t = -4.0, allow_negative_numbers = true)]
    beam_threshold: f64,
    /// Also write fast_align's statistics log (PREFIX.<dir>.log).
    #[arg(long, requires = "output")]
    log: bool,
    /// No progress output.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Args)]
struct AlignArgs {
    /// MODEL SRC [TGT], or with --table: SRC [TGT]. SRC alone is a `source ||| target` file.
    /// For a model trained on subword pieces this binary has no tokeniser: the text must already
    /// be segmented as in training, and the output links are between words (projected with the
    /// model's marker).
    files: Vec<PathBuf>,
    /// Output file; standard output if omitted.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Symmetrisation heuristic (needs a model of both directions); otherwise one direction.
    #[arg(short, long)]
    symmetrize: Option<Method>,
    /// Which direction to write without --symmetrize (with --table: which one the table is
    /// for).
    #[arg(long, value_enum, default_value = "fwd")]
    direction: Dirs,
    /// Append ` ||| score` to each line.
    #[arg(long)]
    scores: bool,
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    /// Align with a table written by fast_align's `-p` (or `export-table`) instead of a saved
    /// model, and print fast_align's `-f` format (`src ||| tgt ||| links ||| score`).
    #[arg(long)]
    table: Option<PathBuf>,
    #[command(flatten)]
    model: ModelArgs,
    /// Mean target length per source word for the score (fast_align `-m`; --table only).
    #[arg(short = 'm', long, default_value_t = 1.0)]
    mean_length_multiplier: f64,
}

#[derive(Args)]
struct SymArgs {
    /// Forward alignments (Pharaoh), one line per sentence pair.
    fwd: PathBuf,
    /// Inverse alignments (Pharaoh, source-target orientation); not needed for `invert`.
    inv: Option<PathBuf>,
    #[arg(short, long, default_value = "grow-diag-final-and")]
    method: Method,
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct ExportArgs {
    model: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "fwd")]
    direction: Dirs,
    #[arg(long, default_value_t = -4.0, allow_negative_numbers = true)]
    beam_threshold: f64,
}

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match cli.command {
        Command::Train(a) => cmd_train(a),
        Command::Align(a) => cmd_align(a),
        Command::Symmetrize(a) => cmd_symmetrize(a),
        Command::ExportTable(a) => cmd_export(a),
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("fastalign: error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn with_ext(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn out_file(p: &Path) -> R<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        1 << 20,
        File::create(p).map_err(|e| format!("{}: {e}", p.display()))?,
    ))
}

fn out_writer(p: &Option<PathBuf>) -> R<Box<dyn Write>> {
    Ok(match p {
        Some(p) => Box::new(out_file(p)?),
        None => Box::new(BufWriter::with_capacity(1 << 20, io::stdout().lock())),
    })
}

fn read_file(p: &Path) -> R<Vec<u8>> {
    let mut v = Vec::new();
    if p == Path::new("-") {
        io::stdin().read_to_end(&mut v)?;
    } else {
        v = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(v)
}

fn read_corpus(input: &Path, tgt: &Option<PathBuf>, max_len: Option<usize>) -> R<Corpus> {
    let opts = LoadOptions {
        max_len,
        ..LoadOptions::default()
    };
    Ok(match tgt {
        Some(t) => Corpus::read_pair(input, t, &opts)?,
        None if input == Path::new("-") => Corpus::from_joined_bytes(&read_file(input)?, &opts),
        None => Corpus::read_joined(input, &opts)?,
    })
}

fn cmd_train(a: TrainArgs) -> R<ExitCode> {
    if !a.symmetrize.is_empty() {
        if a.directions.directions() != Directions::Both {
            return Err("--symmetrize needs --directions both".into());
        }
        if a.output.is_none() {
            return Err("--symmetrize needs --output PREFIX".into());
        }
        if a.symmetrize.contains(&Method::Invert) {
            return Err("`invert` is not a symmetrisation of two directions".into());
        }
    }
    if a.output.is_none() && a.directions.directions() == Directions::Both {
        return Err("both directions need --output PREFIX (or choose --directions fwd|inv)".into());
    }
    if a.model.no_diagonal && a.model.tension != 4.0 {
        return Err("--tension needs the diagonal prior (drop --no-diagonal)".into());
    }
    if !a.no_variational_bayes && a.alpha <= 0.0 {
        return Err("--alpha must be > 0".into());
    }
    let boundary = a.subword.boundary()?;
    if a.subword_output && boundary.is_none() {
        return Err("--subword-output needs --subword-marker or --subword-continuation".into());
    }
    let cfg = TrainConfig {
        train: TrainOptions {
            reverse: false,
            iterations: a.iterations,
            favor_diagonal: !a.model.no_diagonal,
            optimize_tension: !a.no_optimize_tension,
            variational_bayes: !a.no_variational_bayes,
            alpha: a.alpha,
            use_null: !a.model.no_null,
            p0: a.model.p0,
            tension: a.model.tension,
            buffer_size: a.buffer_size,
            print_scores: a.scores,
        },
        directions: a.directions.directions(),
        keep_models: a.save.is_some() || a.table,
        subword: boundary.map(|boundary| SubwordInfo {
            boundary,
            tokenizers: Vec::new(),
            settings: Vec::new(),
        }),
        keep_subword_output: a.subword_output,
        ..TrainConfig::default()
    };
    let mut cfg = cfg;
    if a.no_lookup_index {
        cfg.parallel.dense_index_entries = 0;
    }
    let t0 = Instant::now();
    let corpus = read_corpus(&a.input, &a.tgt, a.max_len)?;
    if !a.quiet {
        eprintln!(
            "read {} lines ({} words in the dictionary)",
            corpus.len(),
            corpus.vocab.len()
        );
    }
    let quiet = a.quiet;
    let progress = move |p: &Progress| {
        if !quiet {
            eprintln!("{p}");
        }
    };
    let trained = with_threads(a.threads, || train(&corpus, &cfg, &progress))?;
    for d in cfg.directions.list() {
        let o = trained.output(d).expect("trained direction");
        match &a.output {
            Some(prefix) => {
                let mut w = out_file(&with_ext(prefix, &format!("{}.pharaoh", d.name())))?;
                w.write_all(&corpus.expand_lines(&o.alignments))?;
                w.flush()?;
                if let Some(raw) = &o.subword_alignments {
                    let mut w =
                        out_file(&with_ext(prefix, &format!("{}.subword.pharaoh", d.name())))?;
                    w.write_all(&corpus.expand_lines(raw))?;
                    w.flush()?;
                }
                if a.log {
                    let mut w = out_file(&with_ext(prefix, &format!("{}.log", d.name())))?;
                    w.write_all(&o.log)?;
                    if a.table {
                        writeln!(
                            w,
                            "conditional probabilities: {}",
                            with_ext(prefix, &format!("{}.table", d.name())).display()
                        )?;
                    }
                    w.flush()?;
                }
                if a.table {
                    let mut w = out_file(&with_ext(prefix, &format!("{}.table", d.name())))?;
                    with_threads(a.threads, || {
                        trained.model.write_table(d, a.beam_threshold, &mut w)
                    })?;
                    w.flush()?;
                }
            }
            None => {
                let mut w = BufWriter::new(io::stdout().lock());
                w.write_all(&corpus.expand_lines(&o.alignments))?;
                w.flush()?;
            }
        }
    }
    if let Some(prefix) = &a.output {
        for &m in &a.symmetrize {
            let text = with_threads(a.threads, || trained.symmetrized(m))?;
            let mut w = out_file(&with_ext(prefix, &format!("{}.pharaoh", m.name())))?;
            w.write_all(&corpus.expand_lines(&text))?;
            w.flush()?;
        }
    }
    if let Some(path) = &a.save {
        trained.model.save(path)?;
    }
    if !a.quiet {
        eprintln!("done in {:.1} s", t0.elapsed().as_secs_f64());
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_align(a: AlignArgs) -> R<ExitCode> {
    if a.table.is_some() {
        return align_with_table(&a);
    }
    let (model_path, input, tgt) = match a.files.as_slice() {
        [m, s] => (m, s, None),
        [m, s, t] => (m, s, Some(t.clone())),
        _ => return Err("usage: fastalign align MODEL SRC [TGT]".into()),
    };
    let model = Model::load(model_path)?;
    let corpus = read_corpus(input, &tgt, None)?;
    let text = with_threads(a.threads, || -> Result<Vec<u8>, String> {
        if let Some(m) = a.symmetrize {
            if m == Method::Invert {
                return Err("`invert` is not a symmetrisation of two directions".into());
            }
            if model.fwd.is_none() || model.inv.is_none() {
                return Err("--symmetrize needs a model of both directions".into());
            }
            let f = model
                .align_words(&corpus, Direction::Forward, false)
                .map_err(|e| e.to_string())?;
            let i = model
                .align_words(&corpus, Direction::Inverse, false)
                .map_err(|e| e.to_string())?;
            return fastalign_core::pipeline::symmetrize_pharaoh(&f, &i, m);
        }
        let d = match a.direction {
            Dirs::Inv => Direction::Inverse,
            _ => Direction::Forward,
        };
        model
            .align_words(&corpus, d, a.scores)
            .map_err(|e| e.to_string())
    })?;
    let mut w = out_writer(&a.output)?;
    w.write_all(&text)?;
    w.flush()?;
    Ok(ExitCode::SUCCESS)
}

/// fast_align's `-f` mode (drop-in): `src ||| tgt ||| links ||| score` lines.
fn align_with_table(a: &AlignArgs) -> R<ExitCode> {
    let (input, tgt) = match a.files.as_slice() {
        [s] => (s, None),
        [s, t] => (s, Some(t.clone())),
        _ => return Err("usage: fastalign align --table TABLE SRC [TGT]".into()),
    };
    if a.symmetrize.is_some() {
        return Err("--symmetrize is not available with --table".into());
    }
    let corpus = read_corpus(input, &tgt, None)?;
    let table_path = a.table.as_ref().unwrap();
    let bytes = std::fs::read(table_path).map_err(|e| format!("{}: {e}", table_path.display()))?;
    let table = TextTable::parse(&bytes, &corpus.vocab);
    let stderr = io::stderr();
    let mut log = BufWriter::new(stderr.lock());
    writeln!(log, "Loaded {} translation parameters.", table.loaded)?;
    let opts = ForceOptions {
        reverse: matches!(a.direction, Dirs::Inv),
        favor_diagonal: !a.model.no_diagonal,
        use_null: !a.model.no_null,
        p0: a.model.p0,
        tension: a.model.tension,
        mean_srclen_multiplier: a.mean_length_multiplier,
    };
    let mut out = out_writer(&a.output)?;
    let res = force_align(&corpus, &table, &opts, &mut log, &mut out)?;
    out.flush()?;
    log.flush()?;
    Ok(match res {
        ForceResult::Done(_) => ExitCode::SUCCESS,
        ForceResult::EmptySide(_) => ExitCode::FAILURE,
    })
}

fn cmd_symmetrize(a: SymArgs) -> R<ExitCode> {
    let fwd = read_file(&a.fwd)?;
    let inv = match (&a.inv, a.method) {
        (Some(p), m) if m != Method::Invert => Some(read_file(p)?),
        (None, m) if m != Method::Invert => {
            return Err(format!("{m} needs two alignment files").into())
        }
        _ => None,
    };
    let mut w = out_writer(&a.output)?;
    symmetrize_text(&fwd, inv.as_deref(), a.method, &mut w)?;
    w.flush()?;
    Ok(ExitCode::SUCCESS)
}

fn cmd_export(a: ExportArgs) -> R<ExitCode> {
    let model = Model::load(&a.model)?;
    let d = match a.direction {
        Dirs::Inv => Direction::Inverse,
        _ => Direction::Forward,
    };
    let mut w = out_writer(&a.output)?;
    model.write_table(d, a.beam_threshold, &mut w)?;
    w.flush()?;
    Ok(ExitCode::SUCCESS)
}
