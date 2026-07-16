from streamlit.testing.v1 import AppTest


def test_streamlit_app_completes_prediction_flow():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception

    app.number_input(key="tenure").set_value(6)
    app.number_input(key="monthly_charges").set_value(95.0)
    app.selectbox(key="contract").set_value("Month-to-month")
    app.selectbox(key="internet_service").set_value("Fiber optic")
    app.selectbox(key="payment_method").set_value("Electronic check")
    app.button(key="predict").click().run()

    assert not app.exception
    assert app.metric[0].value.endswith("%")
    assert len(app.get("status")) == 1
