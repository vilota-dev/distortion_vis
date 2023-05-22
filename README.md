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


## Todo
- [ ] Parse the file uploader (JSON) and automatically input values
- [ ] Add support for multiple cameras dynamically ([tabs](https://docs.streamlit.io/library/api-reference/layout/st.tabs)?)
- [ ] Add project error plot like [here](https://github.com/punit-bhatt/double-sphere-model/tree/master)
## Resources
- [Camera Model intrinsics and extrinsics](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec)