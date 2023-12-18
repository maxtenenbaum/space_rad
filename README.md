# space_rad

## Description
We are working to design a program to process and analyze LFP data from depth electrodes in rats to best understand how different variables affect DMN activity.

## Getting Started
We are using Conda for package and environment management but you can also use pip if preferred. It is recommended to use pip if using a Windows or Linux based system.

### Setting Up the Conda Environment
To set up the project environment using Conda:

1. Ensure [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed on your system.

2. Clone the repository:
```bash
git clone https://github.com/maxtenenbaum/space_rad.git
cd space_rad
```
3. Create the Conda environment from the environment.yml file:

```bash
conda env create -f environment.yml
```
4. Activate the environment:
```bash
conda activate space_rad_env
```
### Setting Up the Environment Using Pip
If you prefer to use pip, follow these steps after cloning the repository:

1. Ensure that you have Python installed

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages
```bash
pip install -r requirements.txt
```
