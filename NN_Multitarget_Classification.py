# WORKING MODEL FINAL SUBMISSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate
from difflib import get_close_matches

print('\n\n\n\n')
df = pd.read_csv('Task 01\\final_dataset.csv')
df.columns = df.columns.str.strip()

vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_ingredients = vectorizer.fit_transform(df['Ingredients'].apply(lambda x: ' '.join(x[1:-1].split(', '))))

label_encoder_diet = LabelEncoder()
df['Diet'] = label_encoder_diet.fit_transform(df['Diet'])

label_encoder_flavor = LabelEncoder()
df['Flavor'] = label_encoder_flavor.fit_transform(df['Flavor'])

label_encoder_carb_content = LabelEncoder()
df['Carbohydrate_Content'] = label_encoder_carb_content.fit_transform(df['Carbohydrate_Content'])

label_encoder_protein = LabelEncoder()
df['Protein'] = label_encoder_protein.fit_transform(df['Protein'])

X = pd.concat([df[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']], pd.DataFrame(X_ingredients.toarray())], axis=1)
label_encoder_meal = LabelEncoder()
y_meal_type = label_encoder_meal.fit_transform(df['Meal_Type']).astype(int)

label_encoder_cuisine = LabelEncoder()
y_cuisine_type = label_encoder_cuisine.fit_transform(df['Cuisine_Type']).astype(int)

X_train, X_test, y_meal_type_train, y_meal_type_test, y_cuisine_type_train, y_cuisine_type_test = train_test_split(
    X, y_meal_type, y_cuisine_type, test_size=0.2, random_state=42
)

input_layer = Input(shape=(X.shape[1],))
hidden_layer = Dense(64, activation='relu')(input_layer)

# Head for 'Meal_Type' & 'Cuisine_Type'
output_meal_type = Dense(len(label_encoder_meal.classes_), activation='softmax', name='output_meal_type')(hidden_layer)
output_cuisine_type = Dense(len(label_encoder_cuisine.classes_), activation='softmax', name='output_cuisine_type')(hidden_layer)

model = keras.Model(inputs=input_layer, outputs=[output_meal_type, output_cuisine_type])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, [y_meal_type_train, y_cuisine_type_train], epochs=250, batch_size=32, validation_data=(X_test, [y_meal_type_test, y_cuisine_type_test]))

y_pred_meal_type, y_pred_cuisine_type = model.predict(X_test)
y_pred_meal_type = y_pred_meal_type.argmax(axis=1)
y_pred_cuisine_type = y_pred_cuisine_type.argmax(axis=1)

accuracy_meal_type = accuracy_score(y_meal_type_test, y_pred_meal_type)
accuracy_cuisine_type = accuracy_score(y_cuisine_type_test, y_pred_cuisine_type)

f1_meal_type = f1_score(y_meal_type_test, y_pred_meal_type, average='weighted')
f1_cuisine_type = f1_score(y_cuisine_type_test, y_pred_cuisine_type, average='weighted')

print(f'Model Accuracy (Meal Type): {accuracy_meal_type * 100:.2f}%')
print(f'Model Accuracy (Cuisine Type): {accuracy_cuisine_type * 100:.2f}%')
print(f'F1 Score (Meal Type): {f1_meal_type:.2f}')
print(f'F1 Score (Cuisine Type): {f1_cuisine_type:.2f}')

# Creating inverse mappings for label encodings
inverse_mapping_diet = {v: k for k, v in enumerate(label_encoder_diet.inverse_transform(df['Diet'].unique()))}
inverse_mapping_flavor = {v: k for k, v in enumerate(label_encoder_flavor.inverse_transform(df['Flavor'].unique()))}
inverse_mapping_carb_content = {v: k for k, v in enumerate(label_encoder_carb_content.inverse_transform(df['Carbohydrate_Content'].unique()))}
inverse_mapping_protein = {v: k for k, v in enumerate(label_encoder_protein.inverse_transform(df['Protein'].unique()))}

while True:
    new_dish_name = input("Enter the name of the new dish (or type 'exit' to stop testing): ")
    
    if new_dish_name.lower() == 'exit':
        break
    
    # Check if the dish name exists in the dataset
    close_matches = get_close_matches(new_dish_name, df['Name'], n=1, cutoff=0.8)
    
    if close_matches:
        # Checking for the match in the dataset
        existing_dish = df[df['Name'] == close_matches[0]].iloc[0]
        print(existing_dish)
        
        # Decode the parameters from label encodings
        existing_dish_features = {
            'Diet': inverse_mapping_diet[existing_dish['Diet']],
            'Flavor': inverse_mapping_flavor[existing_dish['Flavor']],
            'Carbohydrate_Content': inverse_mapping_carb_content[existing_dish['Carbohydrate_Content']],
            'Protein': inverse_mapping_protein[existing_dish['Protein']]
        }
        
        print("\nExisting Dish Features:")
        for feature, value in existing_dish_features.items():
            print(f"{feature}: {value}")

        existing_dish_ingredients_encoded = vectorizer.transform([' '.join(existing_dish['Ingredients'][1:-1].split(', '))]).toarray()
        
        existing_dish_features = pd.concat([pd.DataFrame(existing_dish_features, index=[0]), pd.DataFrame(existing_dish_ingredients_encoded)], axis=1)

        prediction_meal_type, prediction_cuisine_type = model.predict(existing_dish_features)
        predicted_meal_type = label_encoder_meal.inverse_transform([prediction_meal_type.argmax()])
        predicted_cuisine_type = label_encoder_cuisine.inverse_transform([prediction_cuisine_type.argmax()])

        print(f'\nPredicted Meal Type: {predicted_meal_type[0]}')
        print(f'Predicted Cuisine Type: {predicted_cuisine_type[0]}')
    else:
        new_dish_ingredients = input("Enter the ingredients of the new dish (comma-separated): ").split(', ')
        new_dish_diet = input("Enter the diet of the new dish (Vegetarian/Non-Vegetarian): ")
        new_dish_flavor = input("Enter the flavor of the new dish: ")
        new_dish_carb_content = input("Enter the carbohydrate content of the new dish (Low/Moderate/High): ")
        new_dish_protein = input("Enter whether the new dish contains protein (Yes/No): ")

        new_dish = pd.DataFrame({
            'Diet': [new_dish_diet],
            'Flavor': [new_dish_flavor],
            'Carbohydrate_Content': [new_dish_carb_content],
            'Protein': [new_dish_protein]
        })

        #Encoding the inputs
        new_dish['Diet'] = label_encoder_diet.transform(new_dish['Diet'])
        new_dish['Flavor'] = label_encoder_flavor.transform(new_dish['Flavor'])
        new_dish['Carbohydrate_Content'] = label_encoder_carb_content.transform(new_dish['Carbohydrate_Content'])
        new_dish['Protein'] = label_encoder_protein.transform(new_dish['Protein'])

        new_dish_ingredients_encoded = vectorizer.transform([' '.join(new_dish_ingredients)]).toarray()

        new_dish_features = pd.concat([new_dish, pd.DataFrame(new_dish_ingredients_encoded)], axis=1)

        # Predicting the meal type and cuisine type of the new dish
        prediction_meal_type, prediction_cuisine_type = model.predict(new_dish_features)
        predicted_meal_type = label_encoder_meal.inverse_transform([prediction_meal_type.argmax()])
        predicted_cuisine_type = label_encoder_cuisine.inverse_transform([prediction_cuisine_type.argmax()])

        print(f'\nPredicted Meal Type: {predicted_meal_type[0]}')
        print(f'Predicted Cuisine Type: {predicted_cuisine_type[0]}')