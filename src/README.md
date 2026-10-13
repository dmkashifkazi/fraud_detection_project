Step	Notebook workflow	VS Code implementation
1	Load and understand the dataset =>	preprocessing.py
2	Select features and handle missing values =>	preprocessing.py
3	Split data: 4M training, 1M evaluation, remaining live =>	preprocessing.py
4	Impute missing numerical values and encode type =>	feature_engineering.py
5	Train Decision Tree and Random Forest models =>	model_train.py
6	Evaluate models and calculate metrics =>	model_train.py
7	Save the trained model and feature information =>	models/
8	Predict fraud on new transactions =>	model_prediction.py
9	Expose predictions through an API =>	main.py
10	Document setup and deployment =>	README.md and Dockerfile


Step 1:
#Create Virtual Environment in Git Bash
python -m venv venv
source venv/Scripts/activate

Step 2:
Install dependent packages from requirements.txt

Step 3:
Test databse connection
python test_connection.py

python main.py
uvicorn app:app --reload