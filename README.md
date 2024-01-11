
## prepare environment 
conda env create -f ppg_bgl.yaml
pip install -r requirements.txt
conda create --name <envname> --file requirements.txt

## python main.py
--deep_learning
--tuning
--eda
--best_parameter

    parser.add_argument("--data_path", default = './data')
    parser.add_argument("--config", default = './config/baseline.yaml', action='store')
    parser.add_argument("--best_parameter", default = False, action='store_true')
    parser.add_argument("--eda", default = False, action='store_true')
    parser.add_argument("--tuning", default = False, action='store_true')
    parser.add_argument("--deep_learning", default = False, action='store_true')

