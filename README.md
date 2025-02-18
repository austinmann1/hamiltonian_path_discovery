# Hamiltonian Path Discovery

An intelligent system for discovering and improving Hamiltonian path algorithms through pattern mining and continuous improvement.

## Overview

This project uses large language models and pattern mining to iteratively discover and improve algorithms for finding Hamiltonian paths in graphs. It combines:
- Pattern-based learning from successful solutions
- Continuous improvement through iteration
- Benchmark comparison with classical solvers
- Theoretical insights from graph properties

## Features

### Pattern Mining
- Code pattern extraction and analysis
- Vertex and subpath pattern tracking
- Success rate and performance metrics
- Failure pattern analysis

### Continuous Improvement
- Pattern-informed prompting
- Success pattern reinforcement
- Failure pattern avoidance
- Graph property insights

### Benchmarking
- SATLIB format integration
- Classical solver comparison (NetworkX, backtracking)
- Performance metrics by graph size
- Detailed failure analysis

### Analysis Tools
- Graph property analysis
- Pattern effectiveness metrics
- Solution validation
- Theoretical bound checking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hamiltonian-path-discovery.git
cd hamiltonian-path-discovery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENROUTER_API_KEY="your-api-key"
```

## Usage

### Running Benchmarks
```bash
python -m src.benchmarks.run_comparison
```

### Generating Test Instances
```bash
python -m src.benchmarks.benchmark_generator
```

### Analyzing Results
Results are stored in the `results/` directory:
- Benchmark results: `benchmark_results_*.json`
- Pattern state: `pattern_state_*.json`

## Project Structure

```
hamiltonian_path_discovery/
├── src/
│   ├── benchmarks/          # Benchmark and testing tools
│   ├── pattern_mining/      # Pattern analysis and storage
│   ├── prompting/           # LLM prompt management
│   └── verification/        # Solution validation
├── results/                 # Benchmark and pattern results
├── docs/                    # Documentation
└── tests/                  # Test suite
```

## Development Status

### Completed Features
- [x] SATLIB benchmark integration
- [x] Pattern mining system
- [x] Solution validation
- [x] Classical solver comparison
- [x] Graph property analysis

### In Progress
- [ ] Advanced theoretical insights
- [ ] Pattern visualization
- [ ] Performance optimization
- [ ] Extended graph properties

### Planned Features
- [ ] Multi-model comparison
- [ ] Interactive pattern exploration
- [ ] Automated theorem discovery
- [ ] Parallel solution execution

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NetworkX team for graph algorithms
- OpenRouter for API access
- SATLIB for benchmark format
