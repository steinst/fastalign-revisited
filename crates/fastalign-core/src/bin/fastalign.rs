//! `fastalign`: native command line for training fast_align's model, saving it, forced
//! alignment with a saved model (or with a fast_align `-p` table) and symmetrisation.
//!
//! fast_align's short flags are accepted where they do not clash with the sister tool
//! `gizapp` (`-o`, `-s`, `-t`, `-q` keep gizapp's meaning): see `fastalign train --help`.

use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use fastalign_core::atomic::AtomicFile;
use fastalign_core::cancel::Cancel;
use fastalign_core::corpus::{Corpus, EmptyPairs, InputBytes, LoadOptions};
use fastalign_core::io::{Direction, Model, SubwordInfo};
use fastalign_core::oracle::{force_align, ForceOptions, ForceResult, TextTable, TrainOptions};
use fastalign_core::pipeline::{
    train_with_sink, with_threads, Directions, Finished, Progress, TrainConfig,
};
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

/// What training does with a line pair that has an empty side.
#[derive(Clone, Copy, ValueEnum)]
enum EmptyPairsArg {
    /// Train on it, as fast_align does (byte-identical; its effect on the other pairs is
    /// reported as a warning when parameters end nonfinite).
    Keep,
    /// Drop it before training; its output line is empty.
    Skip,
    /// Fail, listing the line numbers and sides.
    Error,
}

impl From<EmptyPairsArg> for EmptyPairs {
    fn from(a: EmptyPairsArg) -> Self {
        match a {
            EmptyPairsArg::Keep => EmptyPairs::Keep,
            EmptyPairsArg::Skip => EmptyPairs::Skip,
            EmptyPairsArg::Error => EmptyPairs::Error,
        }
    }
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

/// One direction, for commands that write or read a single one (`align`, `export-table`).
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OneDir {
    Fwd,
    Inv,
}

impl OneDir {
    fn direction(self) -> Direction {
        match self {
            OneDir::Fwd => Direction::Forward,
            OneDir::Inv => Direction::Inverse,
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
    /// Starting tension of the diagonal prior (fast_align `-T`): a finite number of magnitude at
    /// least 1e-6 (at 0 the prior's normaliser is 0/0); negative values favour the
    /// anti-diagonal, as in fast_align.
    #[arg(
        short = 'T',
        long,
        default_value_t = 4.0,
        allow_negative_numbers = true
    )]
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
    /// `-` reads standard input. With two files each file is one side: tabs and `|||` inside a
    /// line are ordinary whitespace and words there.
    input: PathBuf,
    /// Target text, line-aligned with INPUT (`-`: standard input).
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
    /// Line pairs with an empty side: `keep` trains on them as fast_align does (an empty source
    /// side makes the forward length ratio infinite and, with tension optimisation, the tension
    /// NaN, so that direction aligns nothing; a blank line makes every score NaN); `skip` drops
    /// them (empty output line, input line numbers kept); `error` fails and lists them.
    #[arg(long, value_enum, default_value = "keep")]
    empty_pairs: EmptyPairsArg,
    /// Symmetrisation heuristics to write (repeatable). Needs both directions and --output.
    #[arg(short = 's', long)]
    symmetrize: Vec<Method>,
    /// Append ` ||| score` to the directional lines: the log of the pair's probability under the
    /// model, summed over all alignments (length term plus, per target word, the log of the sum
    /// over source positions; the best alignment only chooses the printed links).
    #[arg(long)]
    scores: bool,
    /// Do not build the dense lookup index of the busiest table rows (saves 64 MB per
    /// direction, slower E-step and M-step; no output changes).
    #[arg(long)]
    no_lookup_index: bool,
    /// MiB per direction for the opt-in cell-offset cache (default 0: off). The first
    /// E-step stores each line's table cells, later E-steps and decoding read through them:
    /// en-fr large ≈ 1.7 GB per direction for all lines, −25 % time at 1 thread, −20 % at 20.
    /// Lines beyond the budget are looked up as usual; no output changes.
    #[arg(long, value_name = "MB", default_value_t = 0)]
    offset_cache: usize,
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
    /// Beam threshold of the tables (fast_align `-t`): an entry is written when its probability
    /// is at least the row maximum raised to the power `-threshold` (default -4: `max^4`).
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
    /// MODEL SRC [TGT], or with --table: SRC [TGT]. SRC alone is a `source ||| target` file;
    /// SRC or TGT may be `-` (standard input).
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
    direction: OneDir,
    /// Append ` ||| score` to each line: the log of the sentence pair's probability under the
    /// model, summed over all alignments (length term plus, per target word, the log of the
    /// sum over source positions; not the best alignment's probability). One direction only.
    #[arg(long)]
    scores: bool,
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    /// Align with a table written by fast_align's `-p` (or `export-table`) instead of a saved
    /// model, and print fast_align's `-f` format (`src ||| tgt ||| links ||| score`).
    #[arg(long)]
    table: Option<PathBuf>,
    /// The input is already cut into pieces with the tokeniser stored in the model. Required
    /// for a model that stores a tokeniser (this binary cannot run it; the Python CLI can).
    #[arg(long)]
    input_segmented: bool,
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
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
}

#[derive(Args)]
struct ExportArgs {
    model: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "fwd")]
    direction: OneDir,
    /// Keep entries with probability at least `row maximum ^ -threshold` (fast_align `-t`).
    #[arg(long, default_value_t = -4.0, allow_negative_numbers = true)]
    beam_threshold: f64,
    /// Worker threads (0 = all cores). The output never depends on it.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
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

/// Run a command's parallel work (reading the input, training, exporting) on one pool of the
/// requested size; a pool that cannot be built is an error.
fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> R<T> + Send) -> R<T> {
    match with_threads(threads, || f().map_err(|e| e.to_string())) {
        Ok(r) => r.map_err(Into::into),
        Err(e) => Err(e.into()),
    }
}

fn with_ext(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// A path-based output is staged and only replaces the destination in `finish`; standard output
/// streams. Dropping without `finish` (an error) leaves an existing destination untouched.
enum Out {
    File(AtomicFile),
    Stdout(BufWriter<io::Stdout>),
}

impl Write for Out {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match self {
            Out::File(f) => f.write(b),
            Out::Stdout(s) => s.write(b),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Out::File(f) => f.flush(),
            Out::Stdout(s) => s.flush(),
        }
    }
}

impl Out {
    fn finish(self, path: Option<&Path>) -> R<()> {
        match self {
            Out::File(f) => f
                .commit()
                .map_err(|e| format!("{}: {e}", path.map_or(Path::new("?"), |p| p).display()))?,
            Out::Stdout(mut s) => s.flush()?,
        }
        Ok(())
    }
}

fn out_file(p: &Path) -> R<Out> {
    Ok(Out::File(
        AtomicFile::create(p).map_err(|e| format!("{}: {e}", p.display()))?,
    ))
}

fn out_writer(p: &Option<PathBuf>) -> R<Out> {
    Ok(match p {
        Some(p) => out_file(p)?,
        None => Out::Stdout(BufWriter::with_capacity(1 << 20, io::stdout())),
    })
}

fn is_stdin(p: &Path) -> bool {
    p == Path::new("-")
}

/// Standard input can be read only once: at most one of the inputs may be `-`.
fn check_stdin_once<'a>(paths: impl IntoIterator<Item = &'a Path>) -> R<()> {
    if paths.into_iter().filter(|p| is_stdin(p)).count() > 1 {
        return Err("standard input (`-`) can be only one of the input files".into());
    }
    Ok(())
}

/// An input file, or standard input for `-` (read to end of file). Other paths that are not
/// regular files (`/dev/stdin`, FIFOs, process substitutions) are read to end of file too.
fn input_bytes(p: &Path) -> R<InputBytes> {
    Ok(if is_stdin(p) {
        InputBytes::stdin()?
    } else {
        InputBytes::open(p)?
    })
}

fn read_file(p: &Path) -> R<Vec<u8>> {
    let mut v = Vec::new();
    if is_stdin(p) {
        io::stdin().read_to_end(&mut v)?;
    } else {
        v = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(v)
}

/// One `source ||| target` file, or two line-aligned files (the two sides); either may be `-`.
fn read_corpus(
    input: &Path,
    tgt: &Option<PathBuf>,
    max_len: Option<usize>,
    empty_pairs: EmptyPairs,
) -> R<Corpus> {
    let opts = LoadOptions {
        max_len,
        empty_pairs,
        ..LoadOptions::default()
    };
    Ok(match tgt {
        Some(t) => {
            check_stdin_once([input, t.as_path()])?;
            Corpus::from_pair_bytes(&input_bytes(input)?, &input_bytes(t)?, &opts)?
        }
        None => Corpus::from_joined_bytes(&input_bytes(input)?, &opts)?,
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
    if !a.model.no_diagonal {
        fastalign_core::math::check_tension(a.model.tension)?;
    }
    fastalign_core::math::check_alpha(a.alpha, !a.no_variational_bayes)?;
    fastalign_core::math::check_p0(a.model.p0)?;
    fastalign_core::math::check_finite("--beam-threshold", a.beam_threshold)?;
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
        keep_models: a.save.is_some(),
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
    cfg.parallel.offset_cache_bytes = a.offset_cache.saturating_mul(1 << 20);
    in_pool(a.threads, || run_train(&a, &cfg))
}

/// The work of `train` under the command's pool: read, train (tables are written as each
/// direction finishes, so its model is freed before the next one trains), write the rest.
fn run_train(a: &TrainArgs, cfg: &TrainConfig) -> R<ExitCode> {
    let t0 = Instant::now();
    let corpus = read_corpus(&a.input, &a.tgt, a.max_len, a.empty_pairs.into())?;
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
    // staged tables (not published until everything else has been written)
    let mut tables: Vec<(Direction, AtomicFile)> = Vec::new();
    let trained = {
        let mut sink = |f: &Finished<'_>| -> io::Result<()> {
            let Some(prefix) = a.output.as_ref().filter(|_| a.table) else {
                return Ok(());
            };
            let path = with_ext(prefix, &format!("{}.table", f.direction.name()));
            let tag = |e: io::Error| io::Error::new(e.kind(), format!("{}: {e}", path.display()));
            let mut w = AtomicFile::create(&path).map_err(tag)?;
            f.model
                .write_table(f.vocab, a.beam_threshold, &Cancel::default(), &mut w)
                .map_err(tag)?;
            tables.push((f.direction, w));
            Ok(())
        };
        train_with_sink(&corpus, cfg, &progress, &Cancel::default(), Some(&mut sink))?
    };
    // not part of the reference-format output: printed even with --quiet, never in --log files
    for w in trained.warnings(&corpus) {
        eprintln!("fastalign: warning: {w}");
    }
    for d in cfg.directions.list() {
        let o = trained.output(d).expect("trained direction");
        match &a.output {
            Some(prefix) => {
                let mut w = out_file(&with_ext(prefix, &format!("{}.pharaoh", d.name())))?;
                w.write_all(&o.alignments)?;
                w.finish(Some(&with_ext(prefix, &format!("{}.pharaoh", d.name()))))?;
                if let Some(raw) = &o.subword_alignments {
                    let mut w =
                        out_file(&with_ext(prefix, &format!("{}.subword.pharaoh", d.name())))?;
                    w.write_all(raw)?;
                    w.finish(Some(&with_ext(
                        prefix,
                        &format!("{}.subword.pharaoh", d.name()),
                    )))?;
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
                    w.finish(Some(&with_ext(prefix, &format!("{}.log", d.name()))))?;
                }
                if let Some(k) = tables.iter().position(|(td, _)| *td == d) {
                    let (_, w) = tables.remove(k);
                    let path = with_ext(prefix, &format!("{}.table", d.name()));
                    w.commit().map_err(|e| format!("{}: {e}", path.display()))?;
                }
            }
            None => {
                let mut w = BufWriter::new(io::stdout().lock());
                w.write_all(&o.alignments)?;
                w.flush()?;
            }
        }
    }
    if let Some(prefix) = &a.output {
        for &m in &a.symmetrize {
            let path = with_ext(prefix, &format!("{}.pharaoh", m.name()));
            let mut w = out_file(&path)?;
            trained.write_symmetrized(m, &mut w)?;
            w.finish(Some(&path))?;
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
    // every conflict is rejected here, before the model is loaded
    if a.symmetrize.is_some() && a.scores {
        return Err("--scores belongs to one direction; drop --symmetrize".into());
    }
    if a.symmetrize == Some(Method::Invert) {
        return Err("`invert` is not a symmetrisation of two directions".into());
    }
    let (model_path, input, tgt) = match a.files.as_slice() {
        [m, s] => (m, s, None),
        [m, s, t] => (m, s, Some(t.clone())),
        _ => return Err("usage: fastalign align MODEL SRC [TGT]".into()),
    };
    in_pool(a.threads, || run_align(&a, model_path, input, tgt))
}

fn run_align(a: &AlignArgs, model_path: &Path, input: &Path, tgt: Option<PathBuf>) -> R<ExitCode> {
    let model = Model::load(model_path)?;
    if model
        .subword
        .as_ref()
        .is_some_and(|sw| sw.tokenizers.iter().any(|t| !t.is_empty()))
        && !a.input_segmented
    {
        return Err("this model stores a subword tokeniser, which the native binary cannot run: \
                    plain words would be read as pieces and misaligned. Align plain text with \
                    the Python `fastalign align` command (pip install fastalign), or pass --input-segmented \
                    if the input is already cut into pieces with the model's tokeniser"
            .into());
    }
    let corpus = read_corpus(input, &tgt, None, EmptyPairs::Keep)?;
    let text = (|| -> Result<Vec<u8>, String> {
        if let Some(m) = a.symmetrize {
            if m == Method::Invert {
                return Err("`invert` is not a symmetrisation of two directions".into());
            }
            if model.fwd.is_none() || model.inv.is_none() {
                return Err("--symmetrize needs a model of both directions".into());
            }
            let mapped = model.map_corpus(&corpus);
            let f = model
                .align_words_mapped(&mapped, Direction::Forward, false, &Cancel::default())
                .map_err(|e| e.to_string())?;
            let i = model
                .align_words_mapped(&mapped, Direction::Inverse, false, &Cancel::default())
                .map_err(|e| e.to_string())?;
            return fastalign_core::pipeline::symmetrize_pharaoh(&f, &i, m);
        }
        model
            .align_words(
                &corpus,
                a.direction.direction(),
                a.scores,
                &Cancel::default(),
            )
            .map_err(|e| e.to_string())
    })()?;
    let mut w = out_writer(&a.output)?;
    w.write_all(&text)?;
    w.finish(a.output.as_deref())?;
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
    if !a.model.no_diagonal {
        fastalign_core::math::check_tension(a.model.tension)?;
    }
    if a.model.no_diagonal && a.model.tension != 4.0 {
        return Err("--tension needs the diagonal prior (drop --no-diagonal)".into());
    }
    fastalign_core::math::check_p0(a.model.p0)?;
    fastalign_core::math::check_finite("--mean-length-multiplier", a.mean_length_multiplier)?;
    in_pool(a.threads, || run_align_table(a, input, tgt))
}

fn run_align_table(a: &AlignArgs, input: &Path, tgt: Option<PathBuf>) -> R<ExitCode> {
    let corpus = read_corpus(input, &tgt, None, EmptyPairs::Keep)?;
    let table_path = a.table.as_ref().unwrap();
    let bytes = std::fs::read(table_path).map_err(|e| format!("{}: {e}", table_path.display()))?;
    let table = TextTable::parse(&bytes, &corpus.vocab);
    let stderr = io::stderr();
    let mut log = BufWriter::new(stderr.lock());
    writeln!(log, "Loaded {} translation parameters.", table.loaded)?;
    let opts = ForceOptions {
        reverse: a.direction == OneDir::Inv,
        favor_diagonal: !a.model.no_diagonal,
        use_null: !a.model.no_null,
        p0: a.model.p0,
        tension: a.model.tension,
        mean_srclen_multiplier: a.mean_length_multiplier,
    };
    let mut out = out_writer(&a.output)?;
    let res = force_align(
        &corpus,
        &table,
        &opts,
        &Cancel::default(),
        &mut log,
        &mut out,
    )?;
    log.flush()?;
    // an empty side is fast_align's deliberate partial-output case: what was written stays
    out.finish(a.output.as_deref())?;
    Ok(match res {
        ForceResult::Done(_) => ExitCode::SUCCESS,
        ForceResult::EmptySide(_) => ExitCode::FAILURE,
    })
}

fn cmd_symmetrize(a: SymArgs) -> R<ExitCode> {
    if a.method != Method::Invert {
        check_stdin_once([a.fwd.as_path()].into_iter().chain(a.inv.as_deref()))?;
    }
    in_pool(a.threads, || run_symmetrize(&a))
}

fn run_symmetrize(a: &SymArgs) -> R<ExitCode> {
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
    w.finish(a.output.as_deref())?;
    Ok(ExitCode::SUCCESS)
}

fn cmd_export(a: ExportArgs) -> R<ExitCode> {
    fastalign_core::math::check_finite("--beam-threshold", a.beam_threshold)?;
    in_pool(a.threads, || run_export(&a))
}

fn run_export(a: &ExportArgs) -> R<ExitCode> {
    let model = Model::load(&a.model)?;
    let d = a.direction.direction();
    let has = match d {
        Direction::Forward => model.fwd.is_some(),
        Direction::Inverse => model.inv.is_some(),
    };
    if !has {
        return Err(format!("the model has no '{}' direction", d.name()).into());
    }
    let mut w = out_writer(&a.output)?;
    model.write_table(d, a.beam_threshold, &Cancel::default(), &mut w)?;
    w.finish(a.output.as_deref())?;
    Ok(ExitCode::SUCCESS)
}
