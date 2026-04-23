use criterion::{Criterion, criterion_group, criterion_main};

fn bench_optimizer_hot_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_hot_loop");
    group.sample_size(10);

    group.bench_function("v5_2000_iters", |b| {
        b.iter(|| {
            qrode::optimizer::bench_hot_loop(2_000, 5, 12345).expect("bench run should succeed")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_optimizer_hot_loop);
criterion_main!(benches);
