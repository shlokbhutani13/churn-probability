# Dataset

`churn.csv` is the IBM Telco Customer Churn sample used in IBM's archived [customer churn prediction project](https://github.com/IBM/customer-churn-prediction).

The source file contains 7,043 customer records and 21 columns. This project trains on five fields that a user can enter in the Streamlit form:

| Field | Meaning |
| --- | --- |
| `tenure` | Months with the provider |
| `MonthlyCharges` | Current monthly charge |
| `Contract` | Contract duration |
| `InternetService` | Internet service type |
| `PaymentMethod` | Billing payment method |
| `Churn` | Yes/No training target |

The model excludes names and direct identifiers. Do not treat this sample as current operational customer data.
