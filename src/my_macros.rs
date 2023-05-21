#[macro_export]
macro_rules! timer {
    ($label:expr, $expr:expr) => {{
        let start = std::time::Instant::now();
        let result = $expr;
        let duration = start.elapsed();
        info!("{}: time taken: {:?}", $label, duration);
        result
    }};
}
