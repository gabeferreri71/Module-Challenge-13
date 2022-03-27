# Module-Challenge-13: Venture Funding with Deep Learning
## Prepare the data to be used on the neural network model
### 1. We read the applicants_data.csv using the Pandas read_csv function with Path, and review with the .head() function. The read dataframe is then reviewed by applying the .dtypes function.

### 2. To drop the 'EIN' and 'NAME' columns from the dataframe, we assign the dataframe variable to itself, using the .drop() function with internal parameters (columns=["EIN", "NAME"]), and reviewing with the .head() function.

### 3. For encoded variables, we create a new variable and assign it list() with internal parameters of applicant_data_df.dtypes[applicant_data_df.dtypes == "object"].index, which verifies the "object" dtype and sets the index. Next, we create a OneHotEncoder instance with the following code : enc = OneHotEncoder(sparse=False). For the encoded_data variable, we will be able to fit_transform to the encoder instance with the internal parameters of the original dataframe, as follows: encoded_data = enc.fit_transform(applicant_data_df[categorical_variables]). Now having the encoded variables, we need to put them in a dataframe where the columns are the categorical_variables, through the following: 
### encoded_df = pd.DataFrame(
###    encoded_data,
###    columns = enc.get_feature_names(categorical_variables)
### )
### encoded_df.head()

### 4. To add the original numerical variables, we need the original dataframe without the categorical_variables and concatanate with the encoded_df. To do this, the following code was implemented: 
### numerical_df = applicant_data_df.drop(columns = categorical_variables)

### encoded_df = pd.concat([numerical_df, encoded_df], axis=1)
### followed by review with the .head() function.

### 5. For the features and targets dataset, y (target) will be "IS_SUCCESSFUL" and X everything else, as shown:

### y = encoded_df["IS_SUCCESSFUL"]

### y[:5]

### X = encoded_df.drop(columns=["IS_SUCCESSFUL"])

### X.head()

### 6. Our test train split looks like the following: X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

### 7. We first create a standardscalar instance in variable scalar. We then fit the instance using the internal parameter off X_train. For our training in test sets, we need to use the transform function as follows: 
### X_train_scaled = X_scaler.transform(X_train)
### X_test_scaled = X_scaler.transform(X_test)

## Compile and Evaluate a Binary Classification Model Using a Neural Network 

### 1. Note, for the alternative models, steps 1-4 will be similar with slight modifications. We first start with number_input_features, defined as len(X_train.iloc[0]). We next definde the number of neurons in the output layer, in this case 1. For the hidden nodes of layers 1 and 2, the following code was implemented:

### hidden_nodes_layer1 =  (number_input_features + 1) // 2 
### hidden_nodes_layer1

### hidden_nodes_layer2 =  (hidden_nodes_layer1 + 1) // 2
### hidden_nodes_layer2

### this is followed with the Sequential model instance. We now need to add our first two hidden layers and our output layer for the neurons.

### nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))
### nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))
### nn.add(Dense(units=number_output_neurons, activation="relu"))

### followed by applying .summary() to nn. 

### 2. To complie the sequential model, the follwoing code is used:
### nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

### The model is then fitted using 50 epochs through the follwoing code: fit_model = nn.fit(X_train_scaled, y_train, epochs=50).
### fit_model = nn.fit(X_train_scaled, y_train, epochs=50)

### 3. To access model loss and accuracy, we use the .evaluate() function with the following:

### model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
### print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

### Note that verbose prints the full summary as a check before the print statement.

### 4. To save and export the model as a HDF5 file, we simply input the following:

### file_path = Path("Models/AlphabetSoup.h5")
### nn.save(file_path)

## Optimize the Neural Network Model
### Alternative Model 1: this model follows the above model EXACTLY, except the number of epochs was increased to 75 (lead to decrease in accuracy), and was saved under the name AlphabetSoup_A1.h5.

### Alternative Model 2: For this model, a thired hidden layer was added through the code, 
### hidden_nodes_layer3_A2 = (hidden_nodes_layer2_A2 + 1) // 2
### hidden_nodes_layer3_A2

### with the original epoch amounts of 50 used. The resulting model has higher accuracy that both the orignial and A1 models. The model was saved under the name AlphabetSoup_A2.h5. 
