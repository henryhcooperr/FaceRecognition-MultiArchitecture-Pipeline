## Usage

### Command-Line Interface

The system uses a unified command-line interface:

```bash
python run.py <command> [options]
```

Available commands:

1. **Interactive Menu**:
```bash
python run.py interactive
```

2. **Preprocess** data:
```bash
python run.py preprocess
```

3. **Train** a model:
```bash
python run.py train --model-type cnn --epochs 50
```

4. **Evaluate** a model:
```bash
python run.py evaluate --model-type cnn
```

5. **Predict** on a single image:
```bash
python run.py predict --model-type cnn --image-path path/to/image.jpg
```

6. **Tune** hyperparameters:
```bash
python run.py tune --model-type cnn --n-trials 50
```

7. Check GPU availability:
```bash
python run.py check-gpu
```

8. List available trained models:
```bash
python run.py list-models
```

For help on any command:
```bash
python run.py <command> --help
``` 