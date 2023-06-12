# A Robotic Manipulation Approach to Identifying, Picking, and Placing Recyclable Waste
### Final project for MIT's 6.4212, Robotic Manipulation

![recycle](https://user-images.githubusercontent.com/78260876/207277799-11106481-700b-4b1d-87fe-ba678be964dc.gif)

## Usage

1. First, create a python virtual environment, as model training relies on specific package versions.

```bash
python3 -m venv recycle_env --system-site-packages
source recycle_env/bin/activate
```

2. Generate training data using the jupyter notebook 'make_training_data.ipynb'.

3. Train the model using the jupyter notebook 'clutter_maskrcnn_train.ipynb'.

4. Run the main notebook 'recycle.ipynb' to see the system in action.

Please see Figures/GreenBot.pdf for an in-depth description of the system.
