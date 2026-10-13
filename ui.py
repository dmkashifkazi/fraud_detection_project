
import streamlit as st
import requests

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("Financial Fraud Detection")
st.write("Enter transaction details to check for potential fraud.")

# Transaction input form
with st.form("transaction_form"):
    step = st.number_input(
        "Transaction Step",
        min_value=0,
        value=1
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Balance Before",
        min_value=0.0,
        value=5000.0
    )

    newbalanceOrig = st.number_input(
        "Sender Balance After",
        min_value=0.0,
        value=4000.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Balance Before",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "Receiver Balance After",
        min_value=0.0,
        value=1000.0
    )

    submitted = st.form_submit_button("Check Transaction")

# Send transaction to Flask API
if submitted:
    transaction = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    try:
        with st.spinner("Analyzing transaction..."):
            response = requests.post(
                API_URL,
                json=transaction,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]

            if prediction == 1:
                st.error("⚠️ Prediction: Potential Fraud")
            else:
                st.success("✅ Prediction: Not Fraud")

            st.caption(
                "This is a machine-learning model prediction made by Kashif, "
                "not a guarantee of whether fraud occurred."
            )

        else:
            st.error(
                f"API error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to Flask API. "
            "Please make sure the Flask server is running."
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out. Please try again.")

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

    except (KeyError, ValueError) as error:
        st.error(f"Invalid API response: {error}")
