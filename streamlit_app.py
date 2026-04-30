import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Project 4 - Chocolate Sales Dashboard",
    page_icon="🍫",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    sales = pd.read_csv("data/sales.csv", parse_dates=["order_date"])
    products = pd.read_csv("data/products.csv")
    stores = pd.read_csv("data/stores.csv")
    customers = pd.read_csv("data/customers.csv", parse_dates=["join_date"], dayfirst=False)
    calendar = pd.read_csv("data/calendar.csv", parse_dates=["date"])

    df = sales.merge(products, on="product_id", how="left")
    df = df.merge(stores, on="store_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(calendar, left_on="order_date", right_on="date", how="left")

    df["month_name"] = pd.to_datetime(df["order_date"]).dt.strftime("%b")
    df["order_year_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").astype(str)
    df["avg_price_after_discount"] = df["unit_price"] * (1 - df["discount"])
    return df


def build_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = df.dropna(subset=["quantity", "age", "unit_price", "discount"])

    y = model_df["quantity"].astype(float)

    X = model_df[
        [
            "age",
            "unit_price",
            "discount",
            "loyalty_member",
            "cocoa_percent",
            "weight_g",
            "country",
            "category",
            "brand",
            "store_type",
            "month",
            "day_of_week",
        ]
    ].copy()
    return X, y


@st.cache_resource(show_spinner=False)
def train_quantity_model(df: pd.DataFrame):
    X, y = build_model_data(df)

    # Keep training quick for dashboard interactivity.
    sample_size = min(200000, len(X))
    sample_idx = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]

    num_cols = ["age", "unit_price", "discount", "cocoa_percent", "weight_g", "month", "day_of_week"]
    cat_cols = ["loyalty_member", "country", "category", "brand", "store_type"]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


df_all = load_data()
min_date = df_all["order_date"].min().date()
max_date = df_all["order_date"].max().date()

st.title("🍫 Project 4 Bonus Dashboard: Chocolate Sales")
st.caption(
    "Interactive companion to the Project 4 notebook: end-to-end data overview, EDA insights, "
    "and a supervised modeling playground."
)

with st.sidebar:
    st.header("Filters")
    date_range = st.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    countries = st.multiselect(
        "Country",
        sorted(df_all["country"].dropna().unique().tolist()),
        default=sorted(df_all["country"].dropna().unique().tolist()),
    )
    categories = st.multiselect(
        "Category",
        sorted(df_all["category"].dropna().unique().tolist()),
        default=sorted(df_all["category"].dropna().unique().tolist()),
    )
    loyalty_filter = st.multiselect(
        "Loyalty member",
        [0, 1],
        default=[0, 1],
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (df_all["order_date"].dt.date >= start_date)
    & (df_all["order_date"].dt.date <= end_date)
    & (df_all["country"].isin(countries))
    & (df_all["category"].isin(categories))
    & (df_all["loyalty_member"].isin(loyalty_filter))
)
df = df_all.loc[mask].copy()

if df.empty:
    st.warning("No rows match the selected filters. Please broaden your filters.")
    st.stop()

st.subheader("KPI Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", f"{len(df):,}")
c2.metric("Total Revenue", f"${df['revenue'].sum():,.2f}")
c3.metric("Total Profit", f"${df['profit'].sum():,.2f}")
c4.metric("Avg Revenue per Order", f"${df['revenue'].mean():,.2f}")

tab1, tab2, tab3 = st.tabs(
    ["1) Data & Preparation", "2) EDA & Unsupervised", "3) Supervised Model Playground"]
)

with tab1:
    st.markdown("### Data Acquisition & Preparation")
    st.write(
        "Merged sales, products, stores, customers, and calendar tables using the same keys "
        "as the notebook pipeline."
    )
    st.dataframe(
        df[
            [
                "order_id",
                "order_date",
                "product_name",
                "category",
                "country",
                "age",
                "loyalty_member",
                "quantity",
                "unit_price",
                "discount",
                "revenue",
                "profit",
            ]
        ].head(100),
        use_container_width=True,
    )

    dq1, dq2, dq3 = st.columns(3)
    dq1.metric("Missing values", int(df.isna().sum().sum()))
    dq2.metric("Duplicate order IDs", int(df["order_id"].duplicated().sum()))
    dq3.metric("Distinct products", int(df["product_id"].nunique()))

with tab2:
    st.markdown("### Exploratory Data Analysis")
    monthly = (
        df.groupby("order_year_month", as_index=False)["revenue"]
        .sum()
        .sort_values("order_year_month")
    )
    fig_monthly = px.line(
        monthly,
        x="order_year_month",
        y="revenue",
        title="Revenue Trend by Year-Month",
        markers=True,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    left, right = st.columns(2)
    with left:
        top_products = (
            df.groupby("product_name", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(10)
        )
        fig_products = px.bar(
            top_products,
            x="product_name",
            y="revenue",
            title="Top 10 Products by Revenue",
        )
        fig_products.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_products, use_container_width=True)

    with right:
        by_country = (
            df.groupby("country", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        )
        fig_country = px.bar(by_country, x="country", y="revenue", title="Revenue by Country")
        st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("### Unsupervised Lens (KMeans-style view from notebook features)")
    unsup = df[["age", "quantity", "revenue"]].dropna().sample(min(20000, len(df)), random_state=42)
    quantiles = pd.qcut(unsup["revenue"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    unsup_plot = unsup.assign(revenue_bucket=quantiles.astype(str))
    fig_unsup = px.scatter(
        unsup_plot,
        x="age",
        y="quantity",
        color="revenue_bucket",
        opacity=0.6,
        title="Segment View: Age vs Quantity (colored by revenue quartile)",
    )
    st.plotly_chart(fig_unsup, use_container_width=True)

with tab3:
    st.markdown("### Supervised Modeling Playground (Target: `quantity`)")
    st.write(
        "This section trains a fast baseline model (Ridge regression with preprocessing) "
        "on a reproducible sample and allows single-order what-if predictions."
    )

    with st.spinner("Training model (cached after first run)..."):
        model, metrics = train_quantity_model(df_all)

    m1, m2, m3 = st.columns(3)
    m1.metric("Validation RMSE", f"{metrics['rmse']:.3f}")
    m2.metric("Validation MAE", f"{metrics['mae']:.3f}")
    m3.metric("Validation R²", f"{metrics['r2']:.3f}")

    st.markdown("#### Predict quantity for a hypothetical order")
    p1, p2, p3 = st.columns(3)
    age = p1.slider("Age", min_value=18, max_value=80, value=35)
    unit_price = p2.slider("Unit price", min_value=1.0, max_value=50.0, value=12.5, step=0.1)
    discount = p3.slider("Discount", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

    q1, q2, q3 = st.columns(3)
    loyalty_member = q1.selectbox("Loyalty member", [0, 1], index=1)
    country = q2.selectbox("Country", sorted(df_all["country"].dropna().unique().tolist()))
    category = q3.selectbox("Category", sorted(df_all["category"].dropna().unique().tolist()))

    r1, r2, r3 = st.columns(3)
    brand = r1.selectbox("Brand", sorted(df_all["brand"].dropna().unique().tolist()))
    store_type = r2.selectbox("Store type", sorted(df_all["store_type"].dropna().unique().tolist()))
    cocoa_percent = r3.slider("Cocoa %", min_value=40, max_value=100, value=75)

    s1, s2 = st.columns(2)
    weight_g = s1.slider("Weight (g)", min_value=50, max_value=500, value=120)
    month = s2.selectbox("Month", list(range(1, 13)), index=5)
    day_of_week = st.selectbox("Day of week (0=Mon ... 6=Sun)", list(range(0, 7)), index=4)

    predict_df = pd.DataFrame(
        [
            {
                "age": age,
                "unit_price": unit_price,
                "discount": discount,
                "loyalty_member": loyalty_member,
                "cocoa_percent": cocoa_percent,
                "weight_g": weight_g,
                "country": country,
                "category": category,
                "brand": brand,
                "store_type": store_type,
                "month": month,
                "day_of_week": day_of_week,
            }
        ]
    )
    pred_quantity = float(model.predict(predict_df)[0])
    st.success(f"Predicted quantity: **{pred_quantity:.2f} units**")

st.markdown("---")
st.caption(
    "Built as Project 4 optional bonus: interactive dashboard/web app aligned with the assignment workflow."
)
