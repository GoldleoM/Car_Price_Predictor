{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fad16be4-bfc1-4a42-ab93-600bbc00ca71",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import pandas as pd\n",
    "\n",
    "# Load pipeline (model + preprocessing together)\n",
    "pipe= pickle.load(open(\"LinearRegressionModel.pkl\", \"rb\"))\n",
    "st.title(\"🚗 Car Price Prediction\")\n",
    "\n",
    "ohe = pipe.named_steps['columntransformer']['onehotencoder']\n",
    "\n",
    "ohe\n",
    "\n",
    "# Assuming order: name, company, fuel_type\n",
    "names = ohe.categories[0]\n",
    "companies = ohe.categories[1]\n",
    "fuel_types = ohe.categories[2]\n",
    "\n",
    "# -------------------------------\n",
    "# UI Inputs\n",
    "# -------------------------------\n",
    "name = st.selectbox(\"Car Name\", names)\n",
    "\n",
    "# Auto-fill company (optional but recommended)\n",
    "company = name.split()[0]\n",
    "st.write(f\"Company: {company}\")\n",
    "\n",
    "year = st.number_input(\"Year\", min_value=1995, max_value=2019)\n",
    "kms_driven = st.number_input(\"KMs Driven\", min_value=0)\n",
    "fuel_type = st.selectbox(\"Fuel Type\", fuel_types)\n",
    "\n",
    "# -------------------------------\n",
    "# Prediction\n",
    "# -------------------------------\n",
    "if st.button(\"Predict Price\"):\n",
    "\n",
    "    input_df = pd.DataFrame({\n",
    "        \"name\": [name],\n",
    "        \"company\": [company],\n",
    "        \"year\": [year],\n",
    "        \"kms_driven\": [kms_driven],\n",
    "        \"fuel_type\": [fuel_type]\n",
    "    })\n",
    "\n",
    "    prediction = pipe.predict(input_df)\n",
    "\n",
    "    st.success(f\"💰 Estimated Price: ₹ {int(prediction[0]):,}\")\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
