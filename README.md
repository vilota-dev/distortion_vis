# Distortion Model Visualiser

- Supports quiver plots for now

## Streamlit UI
- Below is the example for running inside a virtualenv

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
streamlit run stlit.py
```

## Minimal CLI tool

```bash
python cli.py --distortion_model <model> --params <param1> <param2> ...
# Example
python3 cli.py kb4 622 622 965 631 -0.25 -0.0015 0.0007 -0.0002
```


## Supported Distortion Models
The tool currently supports the following distortion models:

- `kb4` (8 parameter version)
- `ds`
- `radtan8`
- `eucm`
