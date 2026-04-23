use clap::Parser;

fn main() {
    let args = qrode::cli::CliArgs::parse();
    if let Err(err) = qrode::cli::run(args) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
