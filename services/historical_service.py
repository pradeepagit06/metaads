import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data")

ADS_FILE = os.path.join(DATA_FOLDER, "ads.csv")
EVENTS_FILE = os.path.join(DATA_FOLDER, "events.csv")
CAMPAIGNS_FILE = os.path.join(DATA_FOLDER, "campaigns.csv")
USERS_FILE = os.path.join(DATA_FOLDER, "users.csv")


def load_data():
    """
    Load all four project datasets.
    """

    ads = pd.read_csv(ADS_FILE)
    events = pd.read_csv(EVENTS_FILE)
    campaigns = pd.read_csv(CAMPAIGNS_FILE)
    users = pd.read_csv(USERS_FILE)

    # Normalize ID columns
    ads["ad_id"] = ads["ad_id"].astype(str).str.strip()
    ads["campaign_id"] = ads["campaign_id"].astype(str).str.strip()

    events["ad_id"] = events["ad_id"].astype(str).str.strip()
    events["user_id"] = events["user_id"].astype(str).str.strip()

    campaigns["campaign_id"] = (
        campaigns["campaign_id"].astype(str).str.strip()
    )

    users["user_id"] = users["user_id"].astype(str).str.strip()

    return ads, events, campaigns, users


def calculate_ad_performance():
    """
    Calculate historical performance for every advertisement.
    """

    ads, events, campaigns, users = load_data()

    # ----------------------------------------
    # EVENT COUNTS FOR EACH AD
    # ----------------------------------------

    event_counts = (
        events
        .pivot_table(
            index="ad_id",
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0
        )
        .reset_index()
    )

    # Make sure every expected event column exists
    expected_events = [
        "Impression",
        "Click",
        "Comment",
        "Like",
        "Share",
        "Purchase"
    ]

    for event in expected_events:
        if event not in event_counts.columns:
            event_counts[event] = 0

    # ----------------------------------------
    # CALCULATE METRICS
    # ----------------------------------------

    event_counts["total_engagement"] = (
        event_counts["Like"]
        + event_counts["Comment"]
        + event_counts["Share"]
    )

    event_counts["ctr"] = (
        event_counts["Click"]
        / event_counts["Impression"].replace(0, pd.NA)
    ) * 100

    event_counts["click_to_purchase_rate"] = (
        event_counts["Purchase"]
        / event_counts["Click"].replace(0, pd.NA)
    ) * 100

    event_counts["purchase_per_impression"] = (
        event_counts["Purchase"]
        / event_counts["Impression"].replace(0, pd.NA)
    ) * 100

    # Replace missing values
    event_counts = event_counts.fillna(0)

    # ----------------------------------------
    # MERGE WITH AD INFORMATION
    # ----------------------------------------

    performance = ads.merge(
        event_counts,
        on="ad_id",
        how="left"
    )

    # Fill ads having no events
    numeric_columns = [
        "Impression",
        "Click",
        "Comment",
        "Like",
        "Share",
        "Purchase",
        "total_engagement",
        "ctr",
        "click_to_purchase_rate",
        "purchase_per_impression"
    ]

    for column in numeric_columns:
        if column in performance.columns:
            performance[column] = performance[column].fillna(0)

    # ----------------------------------------
    # MERGE CAMPAIGN INFORMATION
    # ----------------------------------------

    performance = performance.merge(
        campaigns[
            [
                "campaign_id",
                "name",
                "start_date",
                "end_date",
                "duration_days",
                "total_budget"
            ]
        ],
        on="campaign_id",
        how="left"
    )

    return performance


def get_ad_performance(ad_id):
    """
    Return performance information for one advertisement.
    """

    performance = calculate_ad_performance()

    ad_id = str(ad_id).strip()

    result = performance[
        performance["ad_id"] == ad_id
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


def get_top_ads(limit=10):
    """
    Return top performing advertisements based on CTR.
    """

    performance = calculate_ad_performance()

    result = (
        performance
        .sort_values(
            by="ctr",
            ascending=False
        )
        .head(limit)
    )

    return result.to_dict(orient="records")