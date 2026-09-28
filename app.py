from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

import os
import re
import pytesseract
import pandas as pd

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from flask import send_from_directory
# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ==================================================
# FLASK CONFIGURATION
# ==================================================

app.config["SECRET_KEY"] = "ai-meta-ads-secret-key"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///meta_ads.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

if os.name == "nt":
    # Windows local development
    app.config["UPLOAD_FOLDER"] = os.path.join(
        BASE_DIR,
        "static",
        "uploads"
    )
else:
    # Vercel / Linux
    app.config["UPLOAD_FOLDER"] = os.path.join(
        "/tmp",
        "uploads"
    )

# ==================================================
# DATASET CONFIGURATION
# ==================================================

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

ADS_FILE = os.path.join(
    DATA_FOLDER,
    "ads.csv"
)

EVENTS_FILE = os.path.join(
    DATA_FOLDER,
    "events.csv"
)

CAMPAIGNS_FILE = os.path.join(
    DATA_FOLDER,
    "campaigns.csv"
)

USERS_FILE = os.path.join(
    DATA_FOLDER,
    "users.csv"
)


# ==================================================
# TESSERACT OCR CONFIGURATION
# ==================================================

tesseract_path = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(tesseract_path):

    pytesseract.pytesseract.tesseract_cmd = (
        tesseract_path
    )


# ==================================================
# ALLOWED IMAGE FILE TYPES
# ==================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# ==================================================
# DATABASE
# ==================================================

db = SQLAlchemy(app)


# ==================================================
# USER MODEL
# ==================================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )


# ==================================================
# CREATE DATABASE
# ==================================================

with app.app_context():

    db.create_all()


# ==================================================
# LOAD PROJECT DATASETS
# ==================================================

def load_project_data():

    if not os.path.exists(DATA_FOLDER):

        raise FileNotFoundError(
            "Data folder not found. "
            "Create a folder named 'data' inside "
            "the AI-Meta-Ads project folder."
        )

    required_files = {

        "ads.csv": ADS_FILE,

        "events.csv": EVENTS_FILE,

        "campaigns.csv": CAMPAIGNS_FILE,

        "users.csv": USERS_FILE
    }

    missing_files = []

    for filename, filepath in required_files.items():

        if not os.path.exists(filepath):

            missing_files.append(filename)

    if missing_files:

        raise FileNotFoundError(
            "Missing dataset file(s): "
            + ", ".join(missing_files)
            + ". Put all four CSV files inside the data folder."
        )

    # Read datasets

    ads_df = pd.read_csv(
        ADS_FILE
    )

    events_df = pd.read_csv(
        EVENTS_FILE
    )

    campaigns_df = pd.read_csv(
        CAMPAIGNS_FILE
    )

    users_df = pd.read_csv(
        USERS_FILE
    )

    # ==================================================
    # NORMALIZE ID COLUMNS
    # ==================================================

    if "ad_id" in ads_df.columns:

        ads_df["ad_id"] = (
            ads_df["ad_id"]
            .astype(str)
            .str.strip()
        )

    if "ad_id" in events_df.columns:

        events_df["ad_id"] = (
            events_df["ad_id"]
            .astype(str)
            .str.strip()
        )

    if "user_id" in events_df.columns:

        events_df["user_id"] = (
            events_df["user_id"]
            .astype(str)
            .str.strip()
        )

    if "user_id" in users_df.columns:

        users_df["user_id"] = (
            users_df["user_id"]
            .astype(str)
            .str.strip()
        )

    if "campaign_id" in ads_df.columns:

        ads_df["campaign_id"] = (
            ads_df["campaign_id"]
            .astype(str)
            .str.strip()
        )

    if "campaign_id" in campaigns_df.columns:

        campaigns_df["campaign_id"] = (
            campaigns_df["campaign_id"]
            .astype(str)
            .str.strip()
        )

    return (
        ads_df,
        events_df,
        campaigns_df,
        users_df
    )


# ==================================================
# DATASET SUMMARY
# ==================================================

def get_dataset_summary():

    (
        ads_df,
        events_df,
        campaigns_df,
        users_df
    ) = load_project_data()

    if "event_type" in events_df.columns:

        event_counts = (
            events_df["event_type"]
            .value_counts()
            .to_dict()
        )

    else:

        event_counts = {}

    return {

        "ads": len(ads_df),

        "events": len(events_df),

        "campaigns": len(campaigns_df),

        "users": len(users_df),

        "event_types": event_counts
    }


# ==================================================
# HISTORICAL ADVERTISEMENT INTELLIGENCE
# ==================================================

def calculate_ad_performance():

    """
    Calculate historical performance for every advertisement
    using the actual events dataset.
    """

    (
        ads_df,
        events_df,
        campaigns_df,
        users_df
    ) = load_project_data()

    # ==================================================
    # COUNT EVENTS FOR EACH AD
    # ==================================================

    event_counts = (
        events_df
        .pivot_table(
            index="ad_id",
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0
        )
        .reset_index()
    )

    # ==================================================
    # ENSURE ALL EVENT COLUMNS EXIST
    # ==================================================

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

    # ==================================================
    # TOTAL ENGAGEMENT
    # ==================================================

    event_counts["total_engagement"] = (
        event_counts["Like"]
        + event_counts["Comment"]
        + event_counts["Share"]
    )

    # ==================================================
    # CTR
    # ==================================================

    event_counts["ctr"] = (
        event_counts["Click"]
        / event_counts["Impression"].replace(0, pd.NA)
    ) * 100

    # ==================================================
    # CLICK TO PURCHASE RATE
    # ==================================================

    event_counts["click_to_purchase_rate"] = (
        event_counts["Purchase"]
        / event_counts["Click"].replace(0, pd.NA)
    ) * 100

    # ==================================================
    # PURCHASE PER IMPRESSION
    # ==================================================

    event_counts["purchase_per_impression"] = (
        event_counts["Purchase"]
        / event_counts["Impression"].replace(0, pd.NA)
    ) * 100

    # Replace missing values

    event_counts = event_counts.fillna(0)

    # ==================================================
    # MERGE ADS + EVENT PERFORMANCE
    # ==================================================

    performance = ads_df.merge(
        event_counts,
        on="ad_id",
        how="left"
    )

    # ==================================================
    # FILL ADS WITH NO EVENTS
    # ==================================================

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

            performance[column] = (
                performance[column]
                .fillna(0)
            )

    # ==================================================
    # MERGE CAMPAIGN INFORMATION
    # ==================================================

    performance = performance.merge(

        campaigns_df[
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


# ==================================================
# GET ONE AD PERFORMANCE
# ==================================================

def get_ad_performance(ad_id):

    performance = calculate_ad_performance()

    ad_id = str(ad_id).strip()

    result = performance[
        performance["ad_id"] == ad_id
    ]

    if result.empty:

        return None

    return result.iloc[0].to_dict()


# ==================================================
# GET TOP PERFORMING ADS
# ==================================================

def get_top_ads(limit=10):

    performance = calculate_ad_performance()

    result = (
        performance
        .sort_values(
            by="ctr",
            ascending=False
        )
        .head(limit)
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# CHECK IMAGE FILE EXTENSION
# ==================================================

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()

        in ALLOWED_EXTENSIONS
    )


# ==================================================
# OCR
# ==================================================

def extract_text_from_image(image_path):

    try:

        image = Image.open(
            image_path
        )

        text = pytesseract.image_to_string(
            image
        )

        return text.strip()

    except Exception as e:

        print(
            "OCR ERROR:",
            e
        )

        return ""


# ==================================================
# DETECT AD CATEGORY
# ==================================================

def detect_category(text):

    text_lower = text.lower()

    footwear_words = [

        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "footwear",
        "pair",
        "running"
    ]

    fashion_words = [

        "dress",
        "shirt",
        "fashion",
        "jeans",
        "clothing",
        "wear"
    ]

    food_words = [

        "food",
        "pizza",
        "burger",
        "restaurant",
        "coffee",
        "taste",
        "meal"
    ]

    education_words = [

        "course",
        "education",
        "learn",
        "training",
        "college",
        "academy"
    ]

    technology_words = [

        "software",
        "laptop",
        "mobile",
        "technology",
        "app",
        "digital"
    ]

    if any(
        word in text_lower
        for word in footwear_words
    ):

        return "Footwear / Fashion"

    if any(
        word in text_lower
        for word in fashion_words
    ):

        return "Fashion"

    if any(
        word in text_lower
        for word in food_words
    ):

        return "Food & Beverage"

    if any(
        word in text_lower
        for word in education_words
    ):

        return "Education"

    if any(
        word in text_lower
        for word in technology_words
    ):

        return "Technology"

    return "General Product Advertisement"


# ==================================================
# DETECT PRODUCT
# ==================================================

def detect_product(text):

    text_lower = text.lower()

    if any(
        word in text_lower
        for word in [
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "footwear",
            "pair"
        ]
    ):

        return "Sports / Casual Shoe"

    if any(
        word in text_lower
        for word in [
            "dress",
            "shirt",
            "jeans"
        ]
    ):

        return "Fashion Product"

    if any(
        word in text_lower
        for word in [
            "pizza",
            "burger",
            "coffee"
        ]
    ):

        return "Food Product"

    if any(
        word in text_lower
        for word in [
            "laptop",
            "mobile",
            "phone"
        ]
    ):

        return "Technology Product"

    return "Product not clearly identified"


# ==================================================
# DETECT OFFER
# ==================================================

def detect_offer(text):

    text_lower = text.lower()

    percentage_matches = re.findall(
        r"\b\d{1,3}\s*%",
        text_lower
    )

    if percentage_matches:

        percentages = ", ".join(
            percentage_matches
        )

        if (
            "cash back" in text_lower
            or
            "cashback" in text_lower
        ):

            return (
                f"{percentages} Cashback"
            )

        if "off" in text_lower:

            return (
                f"{percentages} Discount"
            )

        return (
            percentages
            + " Promotional Offer"
        )

    if "cash back" in text_lower:

        return "Cashback Offer"

    if "cashback" in text_lower:

        return "Cashback Offer"

    if "discount" in text_lower:

        return "Discount Offer"

    if "sale" in text_lower:

        return "Sale Promotion"

    return "No clear offer detected"


# ==================================================
# DETECT CREATIVE TYPE
# ==================================================

def detect_creative_type(text):

    text_lower = text.lower()

    if (
        "cashback" in text_lower
        or
        "cash back" in text_lower
        or
        "discount" in text_lower
        or
        "sale" in text_lower
        or
        "%" in text
    ):

        return (
            "Promotional Product Advertisement"
        )

    return "Product-focused Advertisement"


# ==================================================
# GENERATE TARGET AUDIENCE
# ==================================================

def generate_audience(
    category,
    product
):

    if "Footwear" in category:

        return (
            "People interested in footwear, "
            "sportswear and casual fashion"
        )

    if "Fashion" in category:

        return (
            "Fashion-conscious shoppers interested "
            "in clothing and lifestyle products"
        )

    if "Food" in category:

        return (
            "Consumers interested in food, "
            "dining and lifestyle offers"
        )

    if "Education" in category:

        return (
            "Students and learners interested "
            "in education and skill development"
        )

    if "Technology" in category:

        return (
            "Consumers interested in technology "
            "and digital products"
        )

    return (
        "General consumers based on "
        "the advertised product"
    )


# ==================================================
# COMPLETE ADVERTISEMENT ANALYSIS
# ==================================================

def analyze_advertisement(image_path):

    # OCR

    extracted_text = (
        extract_text_from_image(
            image_path
        )
    )

    # Category

    category = detect_category(
        extracted_text
    )

    # Product

    product = detect_product(
        extracted_text
    )

    # Offer

    offer = detect_offer(
        extracted_text
    )

    # Creative type

    creative_type = (
        detect_creative_type(
            extracted_text
        )
    )

    # Audience

    audience = generate_audience(
        category,
        product
    )

    # Historical dataset summary

    try:

        dataset_summary = (
            get_dataset_summary()
        )

    except Exception as e:

        dataset_summary = {

            "ads": 0,

            "events": 0,

            "campaigns": 0,

            "users": 0,

            "event_types": {},

            "error": str(e)
        }

    return {

        "text": extracted_text,

        "category": category,

        "product": product,

        "offer": offer,

        "creative_type": creative_type,

        "audience": audience,

        "historical_data": dataset_summary
    }



# ==================================================
# SIMILAR ADVERTISEMENT INTELLIGENCE
# ==================================================

def calculate_similarity_score(uploaded_analysis, ad_row):
    """
    Calculate a transparent similarity score using only fields
    available in the uploaded advertisement analysis and ads.csv.
    """

    score = 0
    max_score = 0

    uploaded_category = str(
        uploaded_analysis.get("category", "")
    ).lower()

    uploaded_product = str(
        uploaded_analysis.get("product", "")
    ).lower()

    uploaded_offer = str(
        uploaded_analysis.get("offer", "")
    ).lower()

    uploaded_audience = str(
        uploaded_analysis.get("audience", "")
    ).lower()

    uploaded_creative = str(
        uploaded_analysis.get("creative_type", "")
    ).lower()

    target_gender = str(
        ad_row.get("target_gender", "")
    ).lower()

    target_age_group = str(
        ad_row.get("target_age_group", "")
    ).lower()

    target_interests = str(
        ad_row.get("target_interests", "")
    ).lower()

    historical_ad_type = str(
        ad_row.get("ad_type", "")
    ).lower()

    # --------------------------------------------------
    # 1. Interest / category similarity - 30 points
    # --------------------------------------------------
    max_score += 30

    category_keywords = {
        "footwear": [
            "shoe", "shoes", "sneaker", "sneakers",
            "footwear", "sportswear", "running"
        ],
        "fashion": [
            "fashion", "dress", "shirt", "jeans",
            "clothing", "wear"
        ],
        "food": [
            "food", "pizza", "burger", "coffee",
            "restaurant", "meal"
        ],
        "education": [
            "education", "course", "learn",
            "training", "college", "academy"
        ],
        "technology": [
            "technology", "software", "laptop",
            "mobile", "phone", "app", "digital"
        ]
    }

    combined_uploaded_text = (
        uploaded_category
        + " "
        + uploaded_product
        + " "
        + uploaded_offer
        + " "
        + uploaded_audience
    )

    interest_match = False

    for keyword_group in category_keywords.values():
        if any(
            word in combined_uploaded_text
            for word in keyword_group
        ):
            if any(
                word in target_interests
                for word in keyword_group
            ):
                interest_match = True
                break

    # Direct word overlap
    uploaded_words = {
        word.strip(".,/%()-")
        for word in combined_uploaded_text.split()
        if len(word.strip(".,/%()-")) >= 4
    }

    target_words = {
        word.strip(".,/%()-")
        for word in target_interests.split(",")
        if len(word.strip(".,/%()-")) >= 4
    }

    if uploaded_words.intersection(target_words):
        interest_match = True

    if interest_match:
        score += 30

    # --------------------------------------------------
    # 2. Gender similarity - 20 points
    # --------------------------------------------------
    max_score += 20

    if target_gender == "all":
        score += 20
    elif target_gender and target_gender in uploaded_audience:
        score += 20

    # --------------------------------------------------
    # 3. Age-group similarity - 20 points
    # --------------------------------------------------
    max_score += 20

    age_groups = [
        "16-17",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-65"
    ]

    detected_age = None

    for age_group in age_groups:
        if age_group in uploaded_audience:
            detected_age = age_group
            break

    if detected_age and detected_age == target_age_group:
        score += 20
    elif not detected_age:
        # Uploaded OCR does not contain an explicit age group.
        # Give neutral partial weight rather than fabricating a match.
        score += 10

    # --------------------------------------------------
    # 4. Creative/ad type similarity - 20 points
    # --------------------------------------------------
    max_score += 20

    creative_mapping = {
        "image": ["image", "product-focused"],
        "video": ["video"],
        "carousel": ["carousel"],
        "stories": ["stories"],
        "promotional": ["image", "video", "stories", "carousel"]
    }

    creative_match = False

    if "promotional" in uploaded_creative:
        if historical_ad_type in creative_mapping["promotional"]:
            creative_match = True

    for key, values in creative_mapping.items():
        if key in uploaded_creative:
            if historical_ad_type in values:
                creative_match = True

    if creative_match:
        score += 20
    elif historical_ad_type:
        # Small neutral contribution because OCR cannot always
        # determine the exact Meta ad format.
        score += 5

    # --------------------------------------------------
    # 5. Platform compatibility - 10 points
    # --------------------------------------------------
    max_score += 10

    platform = str(
        ad_row.get("ad_platform", "")
    ).lower()

    if platform in ["facebook", "instagram"]:
        score += 10

    similarity = (
        score / max_score
    ) * 100 if max_score else 0

    return round(similarity, 2)


def get_similar_ads(uploaded_analysis, limit=5):
    """
    Find historical advertisements similar to the uploaded ad
    and attach their actual historical performance.
    """

    try:
        data = load_project_data()

        ads_df = data[0]

        performance_df = calculate_ad_performance()

        if ads_df.empty:
            return []

        results = []

        for _, ad_row in ads_df.iterrows():

            similarity = calculate_similarity_score(
                uploaded_analysis,
                ad_row
            )

            ad_id = str(
                ad_row.get("ad_id", "")
            ).strip()

            performance_match = performance_df[
                performance_df["ad_id"].astype(str).str.strip()
                == ad_id
            ]

            if performance_match.empty:
                performance = {}
            else:
                performance = (
                    performance_match.iloc[0].to_dict()
                )

            results.append({
                "ad_id": ad_id,
                "similarity": similarity,

                "platform": str(
                    ad_row.get(
                        "ad_platform",
                        "Unknown"
                    )
                ),

                "ad_type": str(
                    ad_row.get(
                        "ad_type",
                        "Unknown"
                    )
                ),

                "target_gender": str(
                    ad_row.get(
                        "target_gender",
                        "Unknown"
                    )
                ),

                "target_age_group": str(
                    ad_row.get(
                        "target_age_group",
                        "Unknown"
                    )
                ),

                "target_interests": str(
                    ad_row.get(
                        "target_interests",
                        "Unknown"
                    )
                ),

                "impression": int(
                    float(
                        performance.get(
                            "Impression",
                            0
                        )
                    )
                ),

                "click": int(
                    float(
                        performance.get(
                            "Click",
                            0
                        )
                    )
                ),

                "like": int(
                    float(
                        performance.get(
                            "Like",
                            0
                        )
                    )
                ),

                "comment": int(
                    float(
                        performance.get(
                            "Comment",
                            0
                        )
                    )
                ),

                "share": int(
                    float(
                        performance.get(
                            "Share",
                            0
                        )
                    )
                ),

                "purchase": int(
                    float(
                        performance.get(
                            "Purchase",
                            0
                        )
                    )
                ),

                "total_engagement": int(
                    float(
                        performance.get(
                            "total_engagement",
                            0
                        )
                    )
                ),

                "ctr": round(
                    float(
                        performance.get(
                            "ctr",
                            0
                        )
                    ),
                    2
                ),

                "click_to_purchase_rate": round(
                    float(
                        performance.get(
                            "click_to_purchase_rate",
                            0
                        )
                    ),
                    2
                ),

                "purchase_per_impression": round(
                    float(
                        performance.get(
                            "purchase_per_impression",
                            0
                        )
                    ),
                    2
                )
            })

        results.sort(
            key=lambda item: (
                item["similarity"],
                item["ctr"],
                item["purchase"]
            ),
            reverse=True
        )

        return results[:limit]

    except Exception as e:
        print(
            "SIMILAR ADS ERROR:",
            e
        )
        return []
# ==================================================
# BEST RELEASE TIME ENGINE
# ==================================================
def get_best_release_time(uploaded_analysis, limit=5):

    try:

        # ============================================================
        # 1. LOAD PROJECT DATA
        # ============================================================

        (
            ads_df,
            events_df,
            campaigns_df,
            users_df
        ) = load_project_data()


        # ============================================================
        # 2. GET SIMILAR ADVERTISEMENTS
        # ============================================================

        similar_ads = get_similar_ads(
            uploaded_analysis,
            limit=limit
        )

        if not similar_ads:

            return {
                "day": "Not enough data",
                "time": "Not enough data",
                "score": 0.0,
                "impressions": 0,
                "clicks": 0,
                "engagement": 0,
                "purchases": 0,
                "events_analyzed": 0,
                "reason": (
                    "No historical similar advertisements "
                    "were found."
                )
            }


        # ============================================================
        # 3. EXTRACT SIMILAR AD IDs
        # ============================================================

        similar_ad_ids = []

        for ad in similar_ads:

            ad_id = str(
                ad.get("ad_id", "")
            ).strip()

            if ad_id:
                similar_ad_ids.append(ad_id)


        if not similar_ad_ids:

            return {
                "day": "Not enough data",
                "time": "Not enough data",
                "score": 0.0,
                "impressions": 0,
                "clicks": 0,
                "engagement": 0,
                "purchases": 0,
                "events_analyzed": 0,
                "reason": (
                    "Similar advertisements were found, "
                    "but their advertisement IDs were unavailable."
                )
            }


        # ============================================================
        # 4. PREPARE EVENT DATA
        # ============================================================

        events = events_df.copy()

        if events.empty:

            return {
                "day": "Not enough data",
                "time": "Not enough data",
                "score": 0.0,
                "impressions": 0,
                "clicks": 0,
                "engagement": 0,
                "purchases": 0,
                "events_analyzed": 0,
                "reason": (
                    "Historical event data is not available."
                )
            }


        # ============================================================
        # 5. NORMALIZE AD IDs
        # ============================================================

        events["ad_id"] = (
            events["ad_id"]
            .astype(str)
            .str.strip()
        )


        # ============================================================
        # 6. FILTER SIMILAR ADS
        # ============================================================

        events = events[
            events["ad_id"].isin(similar_ad_ids)
        ].copy()


        if events.empty:

            return {
                "day": "Not enough data",
                "time": "Not enough data",
                "score": 0.0,
                "impressions": 0,
                "clicks": 0,
                "engagement": 0,
                "purchases": 0,
                "events_analyzed": 0,
                "reason": (
                    "No historical event data was found "
                    "for the similar advertisements."
                )
            }


        # ============================================================
        # 7. NORMALIZE EVENT TYPE
        # ============================================================

        events["event_type"] = (
            events["event_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        # ============================================================
        # 8. NORMALIZE DAY
        # ============================================================

        events["day_of_week"] = (
            events["day_of_week"]
            .astype(str)
            .str.strip()
        )


        # ============================================================
        # 9. NORMALIZE TIME
        # ============================================================

        events["time_of_day"] = (
            events["time_of_day"]
            .astype(str)
            .str.strip()
        )


        # ============================================================
        # 10. STANDARDIZE EVENT NAMES
        # ============================================================

        event_mapping = {

            "impression": "Impression",
            "impressions": "Impression",

            "click": "Click",
            "clicks": "Click",

            "like": "Like",
            "likes": "Like",

            "comment": "Comment",
            "comments": "Comment",

            "share": "Share",
            "shares": "Share",

            "purchase": "Purchase",
            "purchases": "Purchase"
        }


        events["event_type"] = (
            events["event_type"]
            .map(event_mapping)
            .fillna("")
        )


        # ============================================================
        # 11. REMOVE UNKNOWN EVENTS
        # ============================================================

        events = events[
            events["event_type"] != ""
        ].copy()


        if events.empty:

            return {
                "day": "Not enough data",
                "time": "Not enough data",
                "score": 0.0,
                "impressions": 0,
                "clicks": 0,
                "engagement": 0,
                "purchases": 0,
                "events_analyzed": 0,
                "reason": (
                    "No recognized event types were found "
                    "for the similar advertisements."
                )
            }


        # ============================================================
        # 12. GROUP BY DAY + TIME
        # ============================================================

        grouped = pd.crosstab(
            [
                events["day_of_week"],
                events["time_of_day"]
            ],
            events["event_type"]
        ).reset_index()


        # ============================================================
        # 13. ENSURE ALL REQUIRED EVENT COLUMNS
        # ============================================================

        expected_events = [
            "Impression",
            "Click",
            "Like",
            "Comment",
            "Share",
            "Purchase"
        ]


        for event in expected_events:

            if event not in grouped.columns:
                grouped[event] = 0


        # ============================================================
        # 14. CONVERT EVENT COUNTS TO NUMERIC
        # ============================================================

        for event in expected_events:

            grouped[event] = pd.to_numeric(
                grouped[event],
                errors="coerce"
            ).fillna(0)


        # ============================================================
        # 15. CALCULATE TOTAL ENGAGEMENT
        # ============================================================

        grouped["total_engagement"] = (
            grouped["Like"]
            + grouped["Comment"]
            + grouped["Share"]
        )


        # ============================================================
        # 16. CALCULATE CTR
        # ============================================================

        grouped["ctr"] = 0.0

        impression_mask = (
            grouped["Impression"] > 0
        )

        grouped.loc[
            impression_mask,
            "ctr"
        ] = (
            grouped.loc[
                impression_mask,
                "Click"
            ]
            /
            grouped.loc[
                impression_mask,
                "Impression"
            ]
        ) * 100


        # ============================================================
        # 17. CALCULATE ENGAGEMENT RATE
        # ============================================================

        grouped["engagement_rate"] = 0.0

        grouped.loc[
            impression_mask,
            "engagement_rate"
        ] = (
            grouped.loc[
                impression_mask,
                "total_engagement"
            ]
            /
            grouped.loc[
                impression_mask,
                "Impression"
            ]
        ) * 100


        # ============================================================
        # 18. CALCULATE PURCHASE RATE
        # ============================================================

        grouped["purchase_rate"] = 0.0

        click_mask = (
            grouped["Click"] > 0
        )

        grouped.loc[
            click_mask,
            "purchase_rate"
        ] = (
            grouped.loc[
                click_mask,
                "Purchase"
            ]
            /
            grouped.loc[
                click_mask,
                "Click"
            ]
        ) * 100


        # ============================================================
        # 19. CALCULATE RELEASE SCORE
        # ============================================================

        grouped["release_score"] = (
            grouped["ctr"] * 0.50
            +
            grouped["engagement_rate"] * 0.30
            +
            grouped["purchase_rate"] * 0.20
        )


        # ============================================================
        # 20. CLEAN ALL CALCULATED VALUES
        # ============================================================

        numeric_columns = [
            "Impression",
            "Click",
            "Like",
            "Comment",
            "Share",
            "Purchase",
            "total_engagement",
            "ctr",
            "engagement_rate",
            "purchase_rate",
            "release_score"
        ]


        for column in numeric_columns:

            grouped[column] = pd.to_numeric(
                grouped[column],
                errors="coerce"
            ).fillna(0.0)


        # ============================================================
        # 21. REMOVE INVALID ROWS
        # ============================================================

        grouped = grouped[
            grouped["Impression"] > 0
        ].copy()


        if grouped.empty:

            return {
                "day": "Not enough data",
                "time": "Not enough data",
                "score": 0.0,
                "impressions": 0,
                "clicks": 0,
                "engagement": 0,
                "purchases": 0,
                "events_analyzed": len(events),
                "reason": (
                    "Historical events were found, "
                    "but no impression data was available."
                )
            }


        # ============================================================
        # 22. SORT BY RELEASE SCORE
        # ============================================================

        grouped = grouped.sort_values(
            by="release_score",
            ascending=False
        ).reset_index(drop=True)


        # ============================================================
        # 23. SELECT BEST DAY + TIME
        # ============================================================

        best = grouped.iloc[0]


        best_day = str(
            best["day_of_week"]
        ).strip()


        best_time = str(
            best["time_of_day"]
        ).strip()


        # ============================================================
        # 24. GET COUNTS
        # ============================================================

        impressions = int(
            float(best["Impression"])
        )


        clicks = int(
            float(best["Click"])
        )


        engagement = int(
            float(best["total_engagement"])
        )


        purchases = int(
            float(best["Purchase"])
        )


        # ============================================================
        # 25. DIRECT SCORE CALCULATION
        # ============================================================
        # Recalculate from selected row.
        # This ensures the returned score is never dependent
        # on dataframe formatting or dtype problems.
        # ============================================================

        if impressions > 0:

            ctr = (
                clicks / impressions
            ) * 100

            engagement_rate = (
                engagement / impressions
            ) * 100

        else:

            ctr = 0.0
            engagement_rate = 0.0


        if clicks > 0:

            purchase_rate = (
                purchases / clicks
            ) * 100

        else:

            purchase_rate = 0.0


        # Final weighted score

        score = (
            (ctr * 0.50)
            +
            (engagement_rate * 0.30)
            +
            (purchase_rate * 0.20)
        )


        # ============================================================
        # 26. ROUND SCORE
        # ============================================================

        score = round(
            float(score),
            2
        )


        # ============================================================
        # 27. CREATE EXPLANATION
        # ============================================================

        reason = (
            "This timing recommendation is derived from "
            "historical activity of similar advertisements. "
            "The release score considers click-through rate, "
            "engagement rate and purchase activity."
        )


        # ============================================================
        # 28. RETURN RESULT
        # ============================================================

        return {

            "day": best_day,

            "time": best_time,

            "score": score,

            "impressions": impressions,

            "clicks": clicks,

            "engagement": engagement,

            "purchases": purchases,

            "events_analyzed": len(events),

            "reason": reason
        }


    # ================================================================
    # ERROR HANDLING
    # ================================================================

    except Exception as e:

        print(
            "ERROR in get_best_release_time:",
            str(e)
        )

        return {

            "day": "Unavailable",

            "time": "Unavailable",

            "score": 0.0,

            "impressions": 0,

            "clicks": 0,

            "engagement": 0,

            "purchases": 0,

            "events_analyzed": 0,

            "reason": (
                "Unable to calculate release timing: "
                + str(e)
            )
        }
    # ==================================================
# BEST RELEASE TIME PAGE
# ==================================================

@app.route("/best-release-time/<filename>")
def best_release_time(filename):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    image_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    if not os.path.exists(image_path):

        return "Advertisement image not found!"


    try:

        # ==========================================
        # ANALYZE UPLOADED AD
        # ==========================================

        analysis = analyze_advertisement(

            image_path

        )


        # ==========================================
        # GET BEST RELEASE TIME
        # ==========================================

        recommendation = get_best_release_time(

            analysis,

            limit=5

        )


        # ==========================================
        # RENDER PAGE
        # ==========================================

        return render_template(

            "best_time.html",

            filename=filename,

            analysis=analysis,

            recommendation=recommendation

        )


    except Exception as e:

        return f"""

        <h2>Best Release Time Error</h2>

        <p>{str(e)}</p>

        """
    # ==================================================
# TREND + REGIONAL INTELLIGENCE
# ==================================================

def get_trend_regional_intelligence(
    uploaded_analysis,
    limit=5
):

    try:

        # ==================================================
        # LOAD DATASETS
        # ==================================================

        (
            ads_df,
            events_df,
            campaigns_df,
            users_df
        ) = load_project_data()


        # ==================================================
        # FIND SIMILAR ADS
        # ==================================================

        similar_ads = get_similar_ads(
            uploaded_analysis,
            limit=limit
        )


        if not similar_ads:

            return {

                "trend": [],

                "regional": [],

                "top_country": "Not enough data",

                "top_location": "Not enough data",

                "events_analyzed": 0

            }


        # ==================================================
        # GET SIMILAR AD IDs
        # ==================================================

        similar_ad_ids = [

            str(ad.get("ad_id", "")).strip()

            for ad in similar_ads

            if str(ad.get("ad_id", "")).strip()

        ]


        # ==================================================
        # FILTER EVENTS
        # ==================================================

        events = events_df.copy()

        events["ad_id"] = (
            events["ad_id"]
            .astype(str)
            .str.strip()
        )


        events = events[
            events["ad_id"].isin(
                similar_ad_ids
            )
        ].copy()


        # ==================================================
        # TREND ANALYSIS
        # ==================================================

        events["day_of_week"] = (
            events["day_of_week"]
            .astype(str)
            .str.strip()
        )


        events["time_of_day"] = (
            events["time_of_day"]
            .astype(str)
            .str.strip()
        )


        # ------------------------------------------
        # DAY-WISE EVENT COUNT
        # ------------------------------------------

        day_trend = (
            events
            .groupby("day_of_week")
            .size()
            .reset_index(
                name="events"
            )
        )


        # ------------------------------------------
        # TIME-WISE EVENT COUNT
        # ------------------------------------------

        time_trend = (
            events
            .groupby("time_of_day")
            .size()
            .reset_index(
                name="events"
            )
        )


        # ==================================================
        # ENGAGEMENT BY DAY
        # ==================================================

        engagement_events = events[
            events["event_type"].isin(
                [
                    "Click",
                    "Like",
                    "Comment",
                    "Share",
                    "Purchase"
                ]
            )
        ]


        day_engagement = (
            engagement_events
            .groupby("day_of_week")
            .size()
            .reset_index(
                name="engagement"
            )
        )


        # ==================================================
        # COMBINE DAY TREND
        # ==================================================

        trend_df = day_trend.merge(

            day_engagement,

            on="day_of_week",

            how="left"

        )


        trend_df["engagement"] = (
            trend_df["engagement"]
            .fillna(0)
        )


        trend_df = trend_df.sort_values(

            by="engagement",

            ascending=False

        )


        trend = (
            trend_df
            .head(7)
            .to_dict(
                orient="records"
            )
        )


        # ==================================================
        # REGIONAL ANALYSIS
        # ==================================================

        users = users_df.copy()


        users["user_id"] = (
            users["user_id"]
            .astype(str)
            .str.strip()
        )


        events["user_id"] = (
            events["user_id"]
            .astype(str)
            .str.strip()
        )


        # ==================================================
        # JOIN EVENTS + USERS
        # ==================================================

        regional_data = events.merge(

            users[
                [
                    "user_id",
                    "country",
                    "location",
                    "interests"
                ]
            ],

            on="user_id",

            how="left"

        )


        # ==================================================
        # COUNTRY ANALYSIS
        # ==================================================

        regional_data["country"] = (
            regional_data["country"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )


        country_df = (

            regional_data
            .groupby("country")
            .size()
            .reset_index(
                name="events"
            )

        )


        # ==================================================
        # COUNTRY ENGAGEMENT
        # ==================================================

        regional_engagement = (

            regional_data[
                regional_data["event_type"].isin(
                    [
                        "Click",
                        "Like",
                        "Comment",
                        "Share",
                        "Purchase"
                    ]
                )
            ]

            .groupby("country")
            .size()

            .reset_index(
                name="engagement"
            )

        )


        country_df = country_df.merge(

            regional_engagement,

            on="country",

            how="left"

        )


        country_df["engagement"] = (

            country_df["engagement"]
            .fillna(0)

        )


        # ==================================================
        # SORT COUNTRIES
        # ==================================================

        country_df = country_df.sort_values(

            by="engagement",

            ascending=False

        )


        regional = (

            country_df
            .head(10)
            .to_dict(
                orient="records"
            )

        )


        # ==================================================
        # TOP COUNTRY
        # ==================================================

        if regional:

            top_country = regional[0]["country"]

        else:

            top_country = "Not enough data"


        # ==================================================
        # LOCATION ANALYSIS
        # ==================================================

        regional_data["location"] = (

            regional_data["location"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()

        )


        location_df = (

            regional_data
            .groupby("location")
            .size()
            .reset_index(
                name="events"
            )

        )


        location_df = location_df.sort_values(

            by="events",

            ascending=False

        )


        if not location_df.empty:

            top_location = (
                location_df.iloc[0]["location"]
            )

        else:

            top_location = "Not enough data"


        # ==================================================
        # RETURN RESULTS
        # ==================================================

        return {

            "trend": trend,

            "time_trend": (
                time_trend
                .sort_values(
                    by="events",
                    ascending=False
                )
                .to_dict(
                    orient="records"
                )
            ),

            "regional": regional,

            "top_country": top_country,

            "top_location": top_location,

            "events_analyzed": len(events)

        }


    except Exception as e:

        return {

            "trend": [],

            "time_trend": [],

            "regional": [],

            "top_country": "Unavailable",

            "top_location": "Unavailable",

            "events_analyzed": 0,

            "error": str(e)

        }
    # ==================================================
# TREND + REGIONAL PAGE
# ==================================================

@app.route(
    "/trend-regional/<filename>"
)
def trend_regional(filename):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    image_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    if not os.path.exists(image_path):

        return "Advertisement image not found!"


    try:

        # ==========================================
        # ANALYZE AD
        # ==========================================

        analysis = analyze_advertisement(

            image_path

        )


        # ==========================================
        # TREND + REGIONAL ANALYSIS
        # ==========================================

        intelligence = (
            get_trend_regional_intelligence(

                analysis,

                limit=5

            )
        )


        return render_template(

            "trend_regional.html",

            filename=filename,

            analysis=analysis,

            intelligence=intelligence

        )


    except Exception as e:

        return f"""

        <h2>
            Trend & Regional Analysis Error
        </h2>

        <p>
            {str(e)}
        </p>

        """
    # ============================================================
def get_competitor_intelligence(uploaded_analysis, limit=5):

    try:

        # ==================================================
        # LOAD PROJECT DATA
        # ==================================================

        ads, campaigns, events, users = load_project_data()


        # ==================================================
        # FIND SIMILAR ADVERTISEMENTS
        # ==================================================

        similar_ads = get_similar_ads(
            uploaded_analysis,
            limit=limit
        )


        if not similar_ads:

            return {
                "competitors": [],
                "platform_summary": [],
                "ad_type_summary": [],
                "total_ads_analyzed": 0,
                "insight": "No similar advertisements found."
            }


        # ==================================================
        # GET SIMILAR ADVERTISEMENT IDs
        # ==================================================

        similar_ids = []

        for item in similar_ads:

            if isinstance(item, dict):

                ad_id = item.get("ad_id")

                if ad_id is not None:
                    similar_ids.append(ad_id)


        if not similar_ids:

            return {
                "competitors": [],
                "platform_summary": [],
                "ad_type_summary": [],
                "total_ads_analyzed": 0,
                "insight": (
                    "Similar advertisements found, "
                    "but advertisement IDs are unavailable."
                )
            }


        # ==================================================
        # USE EXISTING HISTORICAL PERFORMANCE FUNCTION
        # ==================================================

        performance_df = calculate_ad_performance()


        if performance_df is None or performance_df.empty:

            return {
                "competitors": [],
                "platform_summary": [],
                "ad_type_summary": [],
                "total_ads_analyzed": 0,
                "insight": (
                    "Historical performance data "
                    "is unavailable."
                )
            }


        # ==================================================
        # CHECK ad_id COLUMN
        # ==================================================

        if "ad_id" not in performance_df.columns:

            return {
                "competitors": [],
                "platform_summary": [],
                "ad_type_summary": [],
                "total_ads_analyzed": 0,
                "insight": (
                    "Historical performance data "
                    "does not contain ad_id."
                ),
                "error": "ad_id missing from performance data"
            }


        # ==================================================
        # FILTER SIMILAR ADS
        # ==================================================

        competitor_df = performance_df[
            performance_df["ad_id"].isin(similar_ids)
        ].copy()


        if competitor_df.empty:

            return {
                "competitors": [],
                "platform_summary": [],
                "ad_type_summary": [],
                "total_ads_analyzed": 0,
                "insight": (
                    "No historical performance found "
                    "for similar advertisements."
                )
            }


        # ==================================================
        # REQUIRED PERFORMANCE COLUMNS
        # ==================================================

        required_columns = [
            "Impression",
            "Click",
            "Like",
            "Comment",
            "Share",
            "Purchase"
        ]


        for column in required_columns:

            if column not in competitor_df.columns:

                competitor_df[column] = 0


        # ==================================================
        # CALCULATE CTR
        # ==================================================

        competitor_df["CTR"] = (
            competitor_df["Click"]
            / competitor_df["Impression"].replace(0, 1)
        ) * 100


        # ==================================================
        # CALCULATE TOTAL ENGAGEMENT
        # ==================================================

        competitor_df["total_engagement"] = (
            competitor_df["Like"]
            + competitor_df["Comment"]
            + competitor_df["Share"]
        )


        # ==================================================
        # CALCULATE ENGAGEMENT RATE
        # ==================================================

        competitor_df["engagement_rate"] = (
            competitor_df["total_engagement"]
            / competitor_df["Impression"].replace(0, 1)
        ) * 100


        # ==================================================
        # CALCULATE PURCHASE RATE
        # ==================================================

        competitor_df["purchase_rate"] = (
            competitor_df["Purchase"]
            / competitor_df["Click"].replace(0, 1)
        ) * 100


        # ==================================================
        # COMPETITIVE SCORE
        # ==================================================
        #
        # CTR             -> 50%
        # Engagement Rate -> 30%
        # Purchase Rate   -> 20%
        #
        # ==================================================

        competitor_df["competitive_score"] = (

            competitor_df["CTR"] * 0.50

            + competitor_df["engagement_rate"] * 0.30

            + competitor_df["purchase_rate"] * 0.20
        )


        # ==================================================
        # SORT BY COMPETITIVE SCORE
        # ==================================================

        competitor_df = competitor_df.sort_values(
            "competitive_score",
            ascending=False
        )


        # ==================================================
        # COMPETITOR DETAILS
        # ==================================================

        competitors = []


        for _, row in competitor_df.head(limit).iterrows():

            competitors.append({

                "ad_id": int(
                    row["ad_id"]
                ),

                "platform": row.get(
                    "ad_platform",
                    "Unknown"
                ),

                "ad_type": row.get(
                    "ad_type",
                    "Unknown"
                ),

                "target_gender": row.get(
                    "target_gender",
                    "Unknown"
                ),

                "target_age_group": row.get(
                    "target_age_group",
                    "Unknown"
                ),

                "target_interests": row.get(
                    "target_interests",
                    "Unknown"
                ),

                "impressions": int(
                    row["Impression"]
                ),

                "clicks": int(
                    row["Click"]
                ),

                "purchases": int(
                    row["Purchase"]
                ),

                "ctr": round(
                    float(row["CTR"]),
                    2
                ),

                "engagement_rate": round(
                    float(row["engagement_rate"]),
                    2
                ),

                "purchase_rate": round(
                    float(row["purchase_rate"]),
                    2
                ),

                "competitive_score": round(
                    float(row["competitive_score"]),
                    2
                )
            })


        # ==================================================
        # PLATFORM SUMMARY
        # ==================================================

        platform_summary = []


        if "ad_platform" in competitor_df.columns:

            platform_group = (
                competitor_df
                .groupby("ad_platform")
                .agg({
                    "Impression": "sum",
                    "Click": "sum",
                    "Purchase": "sum"
                })
                .reset_index()
            )


            for _, row in platform_group.iterrows():

                impressions = int(
                    row["Impression"]
                )

                clicks = int(
                    row["Click"]
                )


                if impressions > 0:

                    ctr = (
                        clicks
                        / impressions
                    ) * 100

                else:

                    ctr = 0


                platform_summary.append({

                    "platform": row["ad_platform"],

                    "impressions": impressions,

                    "clicks": clicks,

                    "purchases": int(
                        row["Purchase"]
                    ),

                    "ctr": round(
                        ctr,
                        2
                    )
                })


        # ==================================================
        # AD TYPE SUMMARY
        # ==================================================

        ad_type_summary = []


        if "ad_type" in competitor_df.columns:

            type_group = (
                competitor_df
                .groupby("ad_type")
                .agg({
                    "Impression": "sum",
                    "Click": "sum",
                    "Purchase": "sum"
                })
                .reset_index()
            )


            for _, row in type_group.iterrows():

                impressions = int(
                    row["Impression"]
                )

                clicks = int(
                    row["Click"]
                )


                if impressions > 0:

                    ctr = (
                        clicks
                        / impressions
                    ) * 100

                else:

                    ctr = 0


                ad_type_summary.append({

                    "ad_type": row["ad_type"],

                    "impressions": impressions,

                    "clicks": clicks,

                    "purchases": int(
                        row["Purchase"]
                    ),

                    "ctr": round(
                        ctr,
                        2
                    )
                })


        # ==================================================
        # GENERATE COMPETITOR INSIGHT
        # ==================================================

        if competitors:

            best = competitors[0]


            insight = (
                "Among historically similar advertisements, "
                f"{best['platform']} "
                f"{best['ad_type']} "
                f"showed a CTR of "
                f"{best['ctr']}% "
                f"and "
                f"{best['purchases']} purchases."
            )

        else:

            insight = (
                "Competitor analysis unavailable."
            )


        # ==================================================
        # FINAL RESULT
        # ==================================================

        return {

            "competitors": competitors,

            "platform_summary": platform_summary,

            "ad_type_summary": ad_type_summary,

            "total_ads_analyzed": len(
                competitor_df
            ),

            "insight": insight
        }


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        return {

            "competitors": [],

            "platform_summary": [],

            "ad_type_summary": [],

            "total_ads_analyzed": 0,

            "insight": (
                "Competitor analysis unavailable."
            ),

            "error": str(e)
        }
def get_cost_estimate(uploaded_analysis, limit=5):

    try:

        # ==========================================
        # LOAD DATA
        # ==========================================

        ads, campaigns, events, users = load_project_data()

        similar_ads = get_similar_ads(
            uploaded_analysis,
            limit=limit
        )

        if not similar_ads:
            return {
                "estimated_cpm": 0,
                "estimated_cpc": 0,
                "estimated_budget": 0,
                "estimated_impressions": 0,
                "ads_used": 0,
                "method": "Historical Similarity-Based Estimate",
                "insight": "No similar advertisements found."
            }

        # ==========================================
        # GET SIMILAR AD IDS
        # ==========================================

        similar_ids = []

        for item in similar_ads:

            if isinstance(item, dict):

                ad_id = item.get("ad_id")

                if ad_id is not None:
                    similar_ids.append(ad_id)

        if not similar_ids:
            return {
                "estimated_cpm": 0,
                "estimated_cpc": 0,
                "estimated_budget": 0,
                "estimated_impressions": 0,
                "ads_used": 0,
                "method": "Historical Similarity-Based Estimate",
                "insight": "Advertisement IDs unavailable."
            }

        # ==========================================
        # HISTORICAL PERFORMANCE
        # ==========================================

        performance_df = calculate_ad_performance()

        if performance_df is None or performance_df.empty:

            return {
                "estimated_cpm": 0,
                "estimated_cpc": 0,
                "estimated_budget": 0,
                "estimated_impressions": 0,
                "ads_used": 0,
                "method": "Historical Similarity-Based Estimate",
                "insight": "Historical performance unavailable."
            }

        # ==========================================
        # FILTER SIMILAR ADS
        # ==========================================

        similar_df = performance_df[
            performance_df["ad_id"].isin(similar_ids)
        ].copy()

        if similar_df.empty:

            return {
                "estimated_cpm": 0,
                "estimated_cpc": 0,
                "estimated_budget": 0,
                "estimated_impressions": 0,
                "ads_used": 0,
                "method": "Historical Similarity-Based Estimate",
                "insight": "No historical data found."
            }

        # ==========================================
        # REQUIRED COLUMNS
        # ==========================================

        for column in [
            "Impression",
            "Click",
            "Purchase"
        ]:

            if column not in similar_df.columns:
                similar_df[column] = 0

        # ==========================================
        # TOTAL METRICS
        # ==========================================

        total_impressions = (
            similar_df["Impression"].sum()
        )

        total_clicks = (
            similar_df["Click"].sum()
        )

        total_purchases = (
            similar_df["Purchase"].sum()
        )

        # ==========================================
        # DATASET-BASED COST MODEL
        # ==========================================
        #
        # These are project-level estimates.
        #
        # Assumed reference:
        # CPM = Rs. 100 per 1000 impressions
        #
        # CPC = CPM / CTR-based conversion
        #
        # ==========================================

        if total_impressions > 0:

            ctr = (
                total_clicks
                / total_impressions
            )

        else:

            ctr = 0

        estimated_cpm = 100.0

        estimated_cpc = (
            estimated_cpm / 1000
        ) / ctr if ctr > 0 else 0

        # ==========================================
        # ESTIMATE FOR 10,000 IMPRESSIONS
        # ==========================================

        target_impressions = 10000

        estimated_budget = (
            target_impressions / 1000
        ) * estimated_cpm

        # ==========================================
        # ROUND VALUES
        # ==========================================

        estimated_cpm = round(
            estimated_cpm,
            2
        )

        estimated_cpc = round(
            estimated_cpc,
            2
        )

        estimated_budget = round(
            estimated_budget,
            2
        )

        # ==========================================
        # INSIGHT
        # ==========================================

        insight = (
            f"Based on {len(similar_df)} historically "
            f"similar advertisements, an estimated "
            f"budget of Rs. {estimated_budget} may be "
            f"used as a project-level reference for "
            f"{target_impressions} impressions."
        )

        # ==========================================
        # FINAL RESULT
        # ==========================================

        return {

            "estimated_cpm": estimated_cpm,

            "estimated_cpc": estimated_cpc,

            "estimated_budget": estimated_budget,

            "estimated_impressions": target_impressions,

            "ads_used": len(similar_df),

            "historical_impressions": int(
                total_impressions
            ),

            "historical_clicks": int(
                total_clicks
            ),

            "historical_purchases": int(
                total_purchases
            ),

            "historical_ctr": round(
                ctr * 100,
                2
            ),

            "method":
                "Historical Similarity-Based Estimate",

            "insight": insight
        }

    except Exception as e:

        return {

            "estimated_cpm": 0,

            "estimated_cpc": 0,

            "estimated_budget": 0,

            "estimated_impressions": 0,

            "ads_used": 0,

            "method":
                "Historical Similarity-Based Estimate",

            "insight":
                "Cost estimation unavailable.",

            "error": str(e)
        }
def get_ad_fatigue_analysis(uploaded_analysis, limit=5):

    try:

        # ==========================================
        # LOAD PROJECT DATA
        # ==========================================

        ads, campaigns, events, users = load_project_data()

        similar_ads = get_similar_ads(
            uploaded_analysis,
            limit=limit
        )

        if not similar_ads:
            return {
                "fatigue_level": "Unknown",
                "fatigue_score": 0,
                "ads_analyzed": 0,
                "trend": "Insufficient historical data",
                "ctr": 0,
                "engagement_rate": 0,
                "purchase_rate": 0,
                "insight": "No similar advertisements found."
            }

        # ==========================================
        # GET SIMILAR AD IDS
        # ==========================================

        similar_ids = []

        for item in similar_ads:

            if isinstance(item, dict):

                ad_id = item.get("ad_id")

                if ad_id is not None:
                    similar_ids.append(ad_id)

        if not similar_ids:
            return {
                "fatigue_level": "Unknown",
                "fatigue_score": 0,
                "ads_analyzed": 0,
                "trend": "Insufficient historical data",
                "ctr": 0,
                "engagement_rate": 0,
                "purchase_rate": 0,
                "insight": "Advertisement IDs unavailable."
            }

        # ==========================================
        # HISTORICAL PERFORMANCE
        # ==========================================

        performance_df = calculate_ad_performance()

        if performance_df is None or performance_df.empty:
            return {
                "fatigue_level": "Unknown",
                "fatigue_score": 0,
                "ads_analyzed": 0,
                "trend": "Insufficient historical data",
                "ctr": 0,
                "engagement_rate": 0,
                "purchase_rate": 0,
                "insight": "Historical performance unavailable."
            }

        # ==========================================
        # FILTER SIMILAR ADS
        # ==========================================

        fatigue_df = performance_df[
            performance_df["ad_id"].isin(similar_ids)
        ].copy()

        if fatigue_df.empty:
            return {
                "fatigue_level": "Unknown",
                "fatigue_score": 0,
                "ads_analyzed": 0,
                "trend": "Insufficient historical data",
                "ctr": 0,
                "engagement_rate": 0,
                "purchase_rate": 0,
                "insight": "No historical performance found."
            }

        # ==========================================
        # REQUIRED COLUMNS
        # ==========================================

        for column in [
            "Impression",
            "Click",
            "Like",
            "Comment",
            "Share",
            "Purchase"
        ]:

            if column not in fatigue_df.columns:
                fatigue_df[column] = 0

        # ==========================================
        # CALCULATE TOTAL VALUES
        # ==========================================

        impressions = fatigue_df["Impression"].sum()
        clicks = fatigue_df["Click"].sum()

        likes = fatigue_df["Like"].sum()
        comments = fatigue_df["Comment"].sum()
        shares = fatigue_df["Share"].sum()

        purchases = fatigue_df["Purchase"].sum()

        engagement = (
            likes
            + comments
            + shares
        )

        # ==========================================
        # CALCULATE PERFORMANCE METRICS
        # ==========================================

        if impressions > 0:

            ctr = (
                clicks / impressions
            ) * 100

            engagement_rate = (
                engagement / impressions
            ) * 100

        else:

            ctr = 0
            engagement_rate = 0

        if clicks > 0:

            purchase_rate = (
                purchases / clicks
            ) * 100

        else:

            purchase_rate = 0

        # ==========================================
        # AD FATIGUE INDICATOR
        # ==========================================
        #
        # Since the project dataset does not contain
        # direct frequency/exposure data, fatigue is
        # treated as a performance-based indicator.
        #
        # Lower CTR + lower engagement + lower
        # purchase activity -> higher fatigue signal.
        #
        # ==========================================

        fatigue_score = 0

        # CTR indicator

        if ctr < 5:

            fatigue_score += 40

        elif ctr < 10:

            fatigue_score += 25

        elif ctr < 15:

            fatigue_score += 10

        else:

            fatigue_score += 0

        # Engagement indicator

        if engagement_rate < 2:

            fatigue_score += 30

        elif engagement_rate < 5:

            fatigue_score += 20

        elif engagement_rate < 8:

            fatigue_score += 10

        else:

            fatigue_score += 0

        # Purchase indicator

        if purchase_rate < 2:

            fatigue_score += 30

        elif purchase_rate < 5:

            fatigue_score += 20

        elif purchase_rate < 8:

            fatigue_score += 10

        else:

            fatigue_score += 0

        # ==========================================
        # LIMIT SCORE
        # ==========================================

        fatigue_score = min(
            fatigue_score,
            100
        )

        # ==========================================
        # FATIGUE LEVEL
        # ==========================================

        if fatigue_score >= 70:

            fatigue_level = "High"

            trend = (
                "Performance indicators suggest "
                "possible creative fatigue."
            )

        elif fatigue_score >= 40:

            fatigue_level = "Moderate"

            trend = (
                "Some performance indicators suggest "
                "moderate fatigue."
            )

        else:

            fatigue_level = "Low"

            trend = (
                "Historical performance indicators "
                "do not show a strong fatigue signal."
            )

        # ==========================================
        # RECOMMENDATION
        # ==========================================

        if fatigue_level == "High":

            recommendation = (
                "Consider refreshing the creative, "
                "testing a new message or changing "
                "the audience strategy."
            )

        elif fatigue_level == "Moderate":

            recommendation = (
                "Consider testing alternative creatives "
                "and monitoring CTR and engagement."
            )

        else:

            recommendation = (
                "Continue monitoring performance while "
                "testing creative variations periodically."
            )

        # ==========================================
        # INSIGHT
        # ==========================================

        insight = (
            f"Historical similar advertisements show "
            f"a CTR of {ctr:.2f}%, engagement rate of "
            f"{engagement_rate:.2f}% and purchase rate "
            f"of {purchase_rate:.2f}%. "
            f"The resulting performance-based fatigue "
            f"indicator is {fatigue_level}."
        )

        # ==========================================
        # FINAL RESULT
        # ==========================================

        return {

            "fatigue_level": fatigue_level,

            "fatigue_score": round(
                fatigue_score,
                2
            ),

            "ads_analyzed": len(
                fatigue_df
            ),

            "impressions": int(
                impressions
            ),

            "clicks": int(
                clicks
            ),

            "engagement": int(
                engagement
            ),

            "purchases": int(
                purchases
            ),

            "ctr": round(
                ctr,
                2
            ),

            "engagement_rate": round(
                engagement_rate,
                2
            ),

            "purchase_rate": round(
                purchase_rate,
                2
            ),

            "trend": trend,

            "recommendation": recommendation,

            "insight": insight,

            "method": (
                "Performance-Based Fatigue Indicator"
            )
        }

    except Exception as e:

        return {

            "fatigue_level": "Unknown",

            "fatigue_score": 0,

            "ads_analyzed": 0,

            "trend": "Analysis unavailable",

            "ctr": 0,

            "engagement_rate": 0,

            "purchase_rate": 0,

            "insight": (
                "Ad fatigue analysis unavailable."
            ),

            "error": str(e)
        }
def get_final_campaign_recommendation(uploaded_analysis, limit=5):

    # ==========================================================
    # SAFE MODULE RUNNER
    # ==========================================================

    def run_module(module_name, function, default):

        try:

            result = function(
                uploaded_analysis,
                limit=limit
            )

            print(
                f"[FINAL] {module_name} module: SUCCESS"
            )

            return result

        except Exception as e:

            print(
                f"[FINAL] {module_name} module ERROR:"
            )

            print(
                type(e).__name__,
                str(e)
            )

            return default

    # ==========================================================
    # DEFAULT VALUES
    # ==========================================================

    performance_default = {
        "predicted_ctr": 0,
        "predicted_engagement_rate": 0,
        "predicted_click_to_purchase_rate": 0
    }

    audience_default = {
        "age_group": "Not available",
        "gender": "Not available",
        "interests": "Not available",
        "platforms": "Not available"
    }

    time_default = {
        "day": "Not available",
        "time": "Not available"
    }

    trend_default = {
        "top_country": "Not available",
        "top_location": "Not available"
    }

    competitor_default = {
        "insight":
            "No competitor insight available."
    }

    cost_default = {
        "estimated_cpm": 0,
        "estimated_cpc": 0,
        "estimated_budget": 0
    }

    fatigue_default = {
        "fatigue_level": "Unknown",
        "fatigue_score": 0,
        "recommendation":
            "Monitor campaign performance."
    }

    # ==========================================================
    # RUN EACH MODULE SEPARATELY
    # ==========================================================

    performance = run_module(
        "Performance Prediction",
        predict_ad_performance,
        performance_default
    )

    audience = run_module(
        "Audience Recommendation",
        get_audience_recommendation,
        audience_default
    )

    best_time = run_module(
        "Best Release Time",
        get_best_release_time,
        time_default
    )

    trend = run_module(
        "Trend + Regional Intelligence",
        get_trend_regional_intelligence,
        trend_default
    )

    competitor = run_module(
        "Competitor Intelligence",
        get_competitor_intelligence,
        competitor_default
    )

    cost = run_module(
        "Cost Estimator",
        get_cost_estimate,
        cost_default
    )

    fatigue = run_module(
        "Ad Fatigue",
        get_ad_fatigue_analysis,
        fatigue_default
    )

    # ==========================================================
    # PERFORMANCE
    # ==========================================================

    predicted_ctr = performance.get(
        "predicted_ctr",
        0
    )

    predicted_engagement = performance.get(
        "predicted_engagement_rate",
        performance.get(
            "predicted_engagement",
            0
        )
    )

    predicted_purchase = performance.get(
        "predicted_click_to_purchase_rate",
        performance.get(
            "predicted_purchase",
            0
        )
    )

    # ==========================================================
    # AUDIENCE
    # ==========================================================

    recommended_age = audience.get(
        "age_group",
        "Not available"
    )

    recommended_gender = audience.get(
        "gender",
        "All"
    )

    recommended_interests = audience.get(
        "interests",
        "Not available"
    )

    recommended_platforms = audience.get(
        "platforms",
        "Not available"
    )

    # ==========================================================
    # FORMAT PLATFORMS
    # ==========================================================

    if isinstance(
        recommended_platforms,
        list
    ):

        recommended_platforms = ", ".join(
            str(platform)
            for platform in recommended_platforms
        )

    # ==========================================================
    # FORMAT INTERESTS
    # ==========================================================

    if isinstance(
        recommended_interests,
        list
    ):

        recommended_interests = ", ".join(
            str(interest)
            for interest in recommended_interests
        )

    # ==========================================================
    # BEST TIME
    # ==========================================================

    recommended_day = best_time.get(
        "day",
        "Not available"
    )

    recommended_time = best_time.get(
        "time",
        "Not available"
    )

    # ==========================================================
    # REGIONAL
    # ==========================================================

    top_country = trend.get(
        "top_country",
        "Not available"
    )

    top_location = trend.get(
        "top_location",
        "Not available"
    )

    if (
        top_country is None
        or top_country == ""
    ):

        top_country = "Not available"

    if (
        top_location is None
        or top_location == ""
    ):

        top_location = "Not available"

    # ==========================================================
    # COST
    # ==========================================================

    estimated_cpm = cost.get(
        "estimated_cpm",
        0
    )

    estimated_cpc = cost.get(
        "estimated_cpc",
        0
    )

    estimated_budget = cost.get(
        "estimated_budget",
        0
    )

    # ==========================================================
    # FATIGUE
    # ==========================================================

    fatigue_level = fatigue.get(
        "fatigue_level",
        "Unknown"
    )

    fatigue_score = fatigue.get(
        "fatigue_score",
        0
    )

    fatigue_action = fatigue.get(
        "recommendation",
        "Monitor campaign performance."
    )

    # ==========================================================
    # CREATIVE ACTION
    # ==========================================================

    if fatigue_level == "High":

        creative_action = (
            "Refresh the advertisement creative "
            "before scaling the campaign."
        )

    elif fatigue_level == "Moderate":

        creative_action = (
            "Test alternative creatives while "
            "monitoring CTR and engagement."
        )

    else:

        creative_action = (
            "Continue monitoring the current "
            "creative and test variations."
        )

    # ==========================================================
    # COMPETITOR
    # ==========================================================

    competitor_insight = competitor.get(
        "insight",
        "No competitor insight available."
    )

    # ==========================================================
    # CAMPAIGN STRATEGY
    # ==========================================================

    campaign_strategy = (
        "Launch the advertisement using the "
        "historically observed audience, platform "
        "and release-time patterns. Start with a "
        "controlled budget and monitor CTR, "
        "engagement and conversion performance "
        "before increasing campaign spend."
    )

    # ==========================================================
    # MONITORING METRICS
    # ==========================================================

    monitoring_points = [

        "CTR",

        "Engagement Rate",

        "Click-to-Purchase Rate",

        "Campaign Cost",

        "Audience Response",

        "Creative Fatigue"

    ]

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    category = uploaded_analysis.get(
        "category",
        "Advertisement"
    )

    product = uploaded_analysis.get(
        "product",
        "Product"
    )

    summary = (
        f"The uploaded {category} advertisement "
        f"for {product} was analyzed using historical "
        f"patterns from similar advertisements. "
        f"The system combines performance prediction, "
        f"audience targeting, release timing, regional "
        f"patterns, competitor comparison, cost "
        f"estimation and creative fatigue indicators "
        f"to generate a campaign planning recommendation."
    )

    # ==========================================================
    # RETURN FINAL RESULT
    # ==========================================================

    return {

        "category":
            category,

        "product":
            product,

        "creative_type":
            uploaded_analysis.get(
                "creative_type",
                "Not available"
            ),

        # Performance
        "predicted_ctr":
            predicted_ctr,

        "predicted_engagement":
            predicted_engagement,

        "predicted_purchase":
            predicted_purchase,

        # Audience
        "recommended_age":
            recommended_age,

        "recommended_gender":
            recommended_gender,

        "recommended_interests":
            recommended_interests,

        "recommended_platforms":
            recommended_platforms,

        # Timing
        "recommended_day":
            recommended_day,

        "recommended_time":
            recommended_time,

        # Region
        "top_country":
            top_country,

        "top_location":
            top_location,

        "regional_target":
            top_country,

        # Cost
        "estimated_cpm":
            estimated_cpm,

        "estimated_cpc":
            estimated_cpc,

        "estimated_budget":
            estimated_budget,

        # Fatigue
        "fatigue_level":
            fatigue_level,

        "fatigue_score":
            fatigue_score,

        "creative_action":
            creative_action,

        "fatigue_action":
            fatigue_action,

        # Competitor
        "competitor_insight":
            competitor_insight,

        # Strategy
        "campaign_strategy":
            campaign_strategy,

        "monitoring_points":
            monitoring_points,

        "summary":
            summary,

        "method":
            "Integrated Historical + ML-Based Recommendation"

    }
def generate_campaign_pdf(filename, analysis, recommendation):

    pdf_folder = os.path.join(
        app.static_folder,
        "reports"
    )

    os.makedirs(
        pdf_folder,
        exist_ok=True
    )

    pdf_filename = (
        os.path.splitext(filename)[0]
        + "_campaign_report.pdf"
    )

    pdf_path = os.path.join(
        pdf_folder,
        pdf_filename
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15
    )

    story = []

    # ==================================================
    # TITLE
    # ==================================================

    story.append(
        Paragraph(
            "AI Meta Advertisement Campaign Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Meta Advertisement Intelligence, "
            "Performance Prediction and Campaign "
            "Recommendation System",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    # ==================================================
    # ADVERTISEMENT PROFILE
    # ==================================================

    story.append(
        Paragraph(
            "1. Advertisement Profile",
            heading_style
        )
    )

    profile_data = [

        ["Category",
         str(recommendation.get(
             "category",
             "Not available"
         ))],

        ["Product",
         str(recommendation.get(
             "product",
             "Not available"
         ))],

        ["Creative Type",
         str(recommendation.get(
             "creative_type",
             "Not available"
         ))]

    ]

    profile_table = Table(
        profile_data,
        colWidths=[150, 330]
    )

    profile_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),

            ("VALIGN", (0, 0), (-1, -1),
             "TOP"),

            ("FONTNAME", (0, 0), (0, -1),
             "Helvetica-Bold"),

            ("FONTNAME", (1, 0), (1, -1),
             "Helvetica")
        ])
    )

    story.append(profile_table)

    # ==================================================
    # SUMMARY
    # ==================================================

    story.append(
        Paragraph(
            "2. AI Campaign Summary",
            heading_style
        )
    )

    story.append(
        Paragraph(
            str(
                recommendation.get(
                    "summary",
                    "Not available"
                )
            ),
            normal_style
        )
    )

    # ==================================================
    # PERFORMANCE
    # ==================================================

    story.append(
        Paragraph(
            "3. Performance Prediction",
            heading_style
        )
    )

    performance_data = [

        ["Metric", "Predicted Value"],

        ["Predicted CTR",
         f"{recommendation.get('predicted_ctr', 0)}%"],

        ["Predicted Engagement",
         f"{recommendation.get('predicted_engagement', 0)}%"],

        ["Click-to-Purchase Rate",
         f"{recommendation.get('predicted_purchase', 0)}%"]

    ]

    performance_table = Table(
        performance_data,
        colWidths=[250, 230]
    )

    performance_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),

            ("FONTNAME", (0, 0), (-1, 0),
             "Helvetica-Bold")
        ])
    )

    story.append(performance_table)

    # ==================================================
    # AUDIENCE
    # ==================================================

    story.append(
        Paragraph(
            "4. Recommended Audience",
            heading_style
        )
    )

    audience_data = [

        ["Parameter", "Recommendation"],

        ["Age Group",
         str(recommendation.get(
             "recommended_age",
             "Not available"
         ))],

        ["Gender",
         str(recommendation.get(
             "recommended_gender",
             "Not available"
         ))],

        ["Interests",
         str(recommendation.get(
             "recommended_interests",
             "Not available"
         ))],

        ["Platforms",
         str(recommendation.get(
             "recommended_platforms",
             "Not available"
         ))]

    ]

    audience_table = Table(
        audience_data,
        colWidths=[180, 300]
    )

    audience_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),

            ("FONTNAME", (0, 0), (-1, 0),
             "Helvetica-Bold"),

            ("VALIGN", (0, 0), (-1, -1),
             "TOP")
        ])
    )

    story.append(audience_table)

    # ==================================================
    # RELEASE TIME
    # ==================================================

    story.append(
        Paragraph(
            "5. Recommended Release Timing",
            heading_style
        )
    )

    timing_data = [

        ["Parameter", "Recommendation"],

        ["Day",
         str(recommendation.get(
             "recommended_day",
             "Not available"
         ))],

        ["Time",
         str(recommendation.get(
             "recommended_time",
             "Not available"
         ))],

        ["Region",
         str(recommendation.get(
             "regional_target",
             "Not available"
         ))]

    ]

    timing_table = Table(
        timing_data,
        colWidths=[180, 300]
    )

    timing_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),

            ("FONTNAME", (0, 0), (-1, 0),
             "Helvetica-Bold")
        ])
    )

    story.append(timing_table)

    # ==================================================
    # COST
    # ==================================================

    story.append(
        Paragraph(
            "6. Campaign Cost Reference",
            heading_style
        )
    )

    cost_data = [

        ["Metric", "Estimated Value"],

        ["CPM",
         f"Rs. {recommendation.get('estimated_cpm', 0)}"],

        ["CPC",
         f"Rs. {recommendation.get('estimated_cpc', 0)}"],

        ["Reference Budget",
         f"Rs. {recommendation.get('estimated_budget', 0)}"]

    ]

    cost_table = Table(
        cost_data,
        colWidths=[250, 230]
    )

    cost_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),

            ("FONTNAME", (0, 0), (-1, 0),
             "Helvetica-Bold")
        ])
    )

    story.append(cost_table)

    # ==================================================
    # FATIGUE
    # ==================================================

    story.append(
        Paragraph(
            "7. Creative Fatigue Analysis",
            heading_style
        )
    )

    fatigue_data = [

        ["Metric", "Result"],

        ["Fatigue Level",
         str(recommendation.get(
             "fatigue_level",
             "Unknown"
         ))],

        ["Fatigue Score",
         f"{recommendation.get('fatigue_score', 0)}/100"],

        ["Recommended Action",
         str(recommendation.get(
             "creative_action",
             "Monitor campaign performance."
         ))]

    ]

    fatigue_table = Table(
        fatigue_data,
        colWidths=[180, 300]
    )

    fatigue_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),

            ("FONTNAME", (0, 0), (-1, 0),
             "Helvetica-Bold"),

            ("VALIGN", (0, 0), (-1, -1),
             "TOP")
        ])
    )

    story.append(fatigue_table)

    # ==================================================
    # COMPETITOR
    # ==================================================

    story.append(
        Paragraph(
            "8. Competitor Intelligence",
            heading_style
        )
    )

    story.append(
        Paragraph(
            str(
                recommendation.get(
                    "competitor_insight",
                    "Not available"
                )
            ),
            normal_style
        )
    )

    # ==================================================
    # CAMPAIGN STRATEGY
    # ==================================================

    story.append(
        Paragraph(
            "9. Recommended Campaign Strategy",
            heading_style
        )
    )

    story.append(
        Paragraph(
            str(
                recommendation.get(
                    "campaign_strategy",
                    "Not available"
                )
            ),
            normal_style
        )
    )

    # ==================================================
    # MONITORING
    # ==================================================

    story.append(
        Paragraph(
            "10. Recommended Monitoring Metrics",
            heading_style
        )
    )

    monitoring_points = recommendation.get(
        "monitoring_points",
        []
    )

    for item in monitoring_points:

        story.append(
            Paragraph(
                "• " + str(item),
                normal_style
            )
        )

    # ==================================================
    # DISCLAIMER
    # ==================================================

    story.append(
        Paragraph(
            "Important Note",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "This report contains historical and "
            "model-based estimates generated from "
            "similar advertisements. Predictions are "
            "not guaranteed future campaign results. "
            "Cost values are project-level reference "
            "estimates and are not live Meta advertising "
            "prices.",
            normal_style
        )
    )

    # ==================================================
    # BUILD PDF
    # ==================================================

    doc.build(story)

    return pdf_filename
# ==================================================
# AUDIENCE RECOMMENDATION ENGINE
# ==================================================

def get_audience_recommendation(uploaded_analysis, limit=5):

    try:
        # Load historical datasets
        (
            ads_df,
            events_df,
            campaigns_df,
            users_df
        ) = load_project_data()

        # Get similar advertisements
        similar_ads = get_similar_ads(
            uploaded_analysis,
            limit=limit
        )

        if not similar_ads:

            return {
                "age_group": "Not enough data",
                "gender": "Not enough data",
                "interests": [],
                "platforms": [],
                "reason": "No historical similar advertisements found.",
                "ads_used": 0
            }

        # ==========================================
        # AGE GROUP
        # ==========================================

        age_counts = {}

        for ad in similar_ads:

            age = str(
                ad.get("target_age_group", "")
            ).strip()

            if age:

                age_counts[age] = (
                    age_counts.get(age, 0) + 1
                )

        recommended_age = "All"

        if age_counts:

            recommended_age = max(
                age_counts,
                key=age_counts.get
            )

        # ==========================================
        # GENDER
        # ==========================================

        gender_counts = {}

        for ad in similar_ads:

            gender = str(
                ad.get("target_gender", "")
            ).strip()

            if gender:

                gender_counts[gender] = (
                    gender_counts.get(gender, 0) + 1
                )

        recommended_gender = "All"

        if gender_counts:

            recommended_gender = max(
                gender_counts,
                key=gender_counts.get
            )

        # ==========================================
        # INTERESTS
        # ==========================================

        interest_counts = {}

        for ad in similar_ads:

            interests = str(
                ad.get("target_interests", "")
            )

            for interest in interests.split(","):

                interest = interest.strip().lower()

                if interest:

                    interest_counts[interest] = (
                        interest_counts.get(interest, 0) + 1
                    )

        # Sort interests by frequency

        sorted_interests = sorted(
            interest_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        recommended_interests = [
            interest.title()
            for interest, count
            in sorted_interests[:5]
        ]

        # ==========================================
        # PLATFORM
        # ==========================================

        platform_counts = {}

        for ad in similar_ads:

            platform = str(
                ad.get("platform", "")
            ).strip()

            if platform:

                platform_counts[platform] = (
                    platform_counts.get(platform, 0) + 1
                )

        recommended_platforms = sorted(
            platform_counts,
            key=platform_counts.get,
            reverse=True
        )

        # ==========================================
        # REASON
        # ==========================================

        reason = (
            "Recommendation is based on the targeting "
            "patterns of historically similar advertisements "
            "and their observed engagement and purchase activity."
        )

        # ==========================================
        # RETURN RESULT
        # ==========================================

        return {

            "age_group": recommended_age,

            "gender": recommended_gender,

            "interests": recommended_interests,

            "platforms": recommended_platforms,

            "reason": reason,

            "ads_used": len(similar_ads)

        }

    except Exception as e:

        return {

            "age_group": "Unavailable",

            "gender": "Unavailable",

            "interests": [],

            "platforms": [],

            "reason": str(e),

            "ads_used": 0

        }
# ==================================================
# SIMILAR ADVERTISEMENT ROUTE
# ==================================================

@app.route("/similar-ads/<filename>")
def similar_ads(filename):

    if "user_id" not in session:
        return redirect(
            url_for("login")
        )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    analysis = analyze_advertisement(
        image_path
    )

    similar_ads_result = get_similar_ads(
        analysis,
        limit=5
    )

    return render_template(
        "similar_ads.html",
        filename=filename,
        analysis=analysis,
        similar_ads=similar_ads_result
    )


# ==================================================
# SIMILAR ADS JSON TEST
# ==================================================

@app.route("/similar-ads-test/<filename>")
def similar_ads_test(filename):

    if "user_id" not in session:
        return redirect(
            url_for("login")
        )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return {
            "status": "error",
            "message": "Advertisement image not found!"
        }, 404

    try:
        analysis = analyze_advertisement(
            image_path
        )

        result = get_similar_ads(
            analysis,
            limit=5
        )

        return {
            "status": "success",
            "message": (
                "Similar advertisement analysis "
                "completed successfully."
            ),
            "uploaded_analysis": analysis,
            "similar_ads": result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }, 500


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==================================================
# REGISTER
# ==================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:

            return (
                "All fields are required."
            )

        existing_user = (
            User.query
            .filter_by(
                email=email
            )
            .first()
        )

        if existing_user:

            return (
                "Email already registered!"
            )

        hashed_password = (
            generate_password_hash(
                password
            )
        )

        new_user = User(

            name=name,

            email=email,

            password=hashed_password
        )

        db.session.add(
            new_user
        )

        db.session.commit()

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ==================================================
# LOGIN
# ==================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = (
            User.query
            .filter_by(
                email=email
            )
            .first()
        )

        if (
            user
            and
            check_password_hash(
                user.password,
                password
            )
        ):

            session["user_id"] = (
                user.id
            )

            session["user_name"] = (
                user.name
            )

            return redirect(
                url_for("dashboard")
            )

        return (
            "Invalid email or password!"
        )

    return render_template(
        "login.html"
    )


# ==================================================
# DASHBOARD
# ==================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(

        "dashboard.html",

        name=session[
            "user_name"
        ]
    )


# ==================================================
# UPLOAD ADVERTISEMENT
# ==================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        if "ad_image" not in request.files:

            return (
                "No advertisement selected!"
            )

        file = request.files[
            "ad_image"
        ]

        if file.filename == "":

            return (
                "Please select an advertisement!"
            )

        if not allowed_file(
            file.filename
        ):

            return (
                "Only PNG, JPG, JPEG "
                "and WEBP images are allowed!"
            )

        os.makedirs(

            app.config[
                "UPLOAD_FOLDER"
            ],

            exist_ok=True
        )

        filename = secure_filename(
            file.filename
        )

        file_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            filename
        )

        file.save(
            file_path
        )

        return render_template(

            "upload.html",

            filename=filename,

            success=True
        )

    return render_template(
        "upload.html"
    )


# ==================================================
# ANALYZE ADVERTISEMENT
# ==================================================

@app.route(
    "/analyze/<filename>"
)
def analyze(filename):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    image_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename
    )

    if not os.path.exists(
        image_path
    ):

        return (
            "Advertisement image not found!"
        )

    analysis = (
        analyze_advertisement(
            image_path
        )
    )

    return render_template(

        "analysis.html",

        filename=filename,

        analysis=analysis
    )


# ==================================================
# DATASET CONNECTION TEST
# ==================================================

@app.route("/data-test")
def data_test():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    try:

        (
            ads_df,
            events_df,
            campaigns_df,
            users_df
        ) = load_project_data()

        if "event_type" in events_df.columns:

            event_types = (

                events_df[
                    "event_type"
                ]

                .value_counts()

                .to_dict()
            )

        else:

            event_types = {}

        return {

            "status": "success",

            "message": (
                "All four datasets "
                "connected successfully."
            ),

            "ads_rows": len(
                ads_df
            ),

            "events_rows": len(
                events_df
            ),

            "campaigns_rows": len(
                campaigns_df
            ),

            "users_rows": len(
                users_df
            ),

            "ads_columns": list(
                ads_df.columns
            ),

            "events_columns": list(
                events_df.columns
            ),

            "campaigns_columns": list(
                campaigns_df.columns
            ),

            "users_columns": list(
                users_df.columns
            ),

            "event_types": event_types
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }, 500


# ==================================================
# HISTORICAL PERFORMANCE TEST
# ==================================================

@app.route("/historical-test")
def historical_test():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    try:

        performance = (
            calculate_ad_performance()
        )

        # Convert first 5 rows to JSON-safe records
        sample = (
            performance
            .head(5)
            .to_json(
                orient="records"
            )
        )

        import json

        sample = json.loads(
            sample
        )

        return {

            "status": "success",

            "message": (
                "Historical advertisement "
                "analysis completed successfully."
            ),

            "total_ads": len(
                performance
            ),

            "sample": sample
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }, 500


# ==================================================
# SINGLE AD PERFORMANCE
# ==================================================

@app.route(
    "/ad-performance/<ad_id>"
)
def ad_performance(ad_id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    try:

        result = get_ad_performance(
            ad_id
        )

        if result is None:

            return {

                "status": "error",

                "message": (
                    f"Advertisement {ad_id} "
                    "not found."
                )
            }, 404

        # Convert pandas/numpy values safely
        result = (
            pd.Series(result)
            .to_json()
        )

        import json

        result = json.loads(
            result
        )

        return {

            "status": "success",

            "ad": result
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }, 500


# ==================================================
# TOP PERFORMING ADS
# ==================================================

@app.route("/top-ads")
def top_ads():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    try:

        limit = request.args.get(
            "limit",
            default=10,
            type=int
        )

        # Keep limit reasonable
        limit = max(
            1,
            min(limit, 50)
        )

        result = get_top_ads(
            limit
        )

        # JSON-safe conversion
        result_json = (
            pd.DataFrame(result)
            .to_json(
                orient="records"
            )
        )

        import json

        result = json.loads(
            result_json
        )

        return {

            "status": "success",

            "message": (
                "Top performing "
                "advertisements retrieved."
            ),

            "count": len(result),

            "ads": result
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }, 500



# ==================================================
# PERFORMANCE PREDICTION FROM SIMILAR HISTORICAL ADS
# ==================================================

def predict_ad_performance(uploaded_analysis, limit=5):
    """
    Estimate uploaded-ad performance from the most similar
    historical advertisements.

    This is a historical similarity-based estimate, not a
    live Meta prediction or guaranteed campaign result.
    """

    similar_ads = get_similar_ads(uploaded_analysis, limit=limit)

    if not similar_ads:
        return {
            "prediction_method": "Historical Similarity-Based Estimate",
            "sample_size": 0,
            "predicted_ctr": 0.0,
            "predicted_engagement_rate": 0.0,
            "predicted_click_to_purchase_rate": 0.0,
            "estimated_purchases_per_1000_impressions": 0.0,
            "confidence": "Low",
            "similar_ads": []
        }

    weights = []
    for ad in similar_ads:
        similarity = max(0.0, float(ad.get("similarity", 0)))
        weights.append(max(similarity, 1.0))

    total_weight = sum(weights)

    def weighted_average(key):
        values = []
        for ad, weight in zip(similar_ads, weights):
            try:
                value = float(ad.get(key, 0))
            except (TypeError, ValueError):
                value = 0.0
            values.append(value * weight)
        return sum(values) / total_weight if total_weight else 0.0

    predicted_ctr = weighted_average("ctr")
    predicted_purchase_rate = weighted_average("click_to_purchase_rate")

    # Engagement rate = (likes + comments + shares) / impressions * 100
    engagement_rates = []
    for ad, weight in zip(similar_ads, weights):
        try:
            impression = float(ad.get("impression", 0))
            engagement = float(ad.get("total_engagement", 0))
            rate = (engagement / impression * 100) if impression > 0 else 0.0
        except (TypeError, ValueError):
            rate = 0.0
        engagement_rates.append((rate, weight))

    predicted_engagement_rate = (
        sum(rate * weight for rate, weight in engagement_rates) / total_weight
        if total_weight else 0.0
    )

    # For an interpretable benchmark, estimate purchases per 1,000 impressions.
    estimated_purchases = (
        1000 * (predicted_ctr / 100) * (predicted_purchase_rate / 100)
    )

    average_similarity = sum(weights) / len(weights) if weights else 0.0

    if average_similarity >= 75:
        confidence = "High"
    elif average_similarity >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "prediction_method": "Historical Similarity-Based Estimate",
        "sample_size": len(similar_ads),
        "predicted_ctr": round(predicted_ctr, 2),
        "predicted_engagement_rate": round(predicted_engagement_rate, 2),
        "predicted_click_to_purchase_rate": round(predicted_purchase_rate, 2),
        "estimated_purchases_per_1000_impressions": round(estimated_purchases, 2),
        "average_similarity": round(average_similarity, 2),
        "confidence": confidence,
        "similar_ads": similar_ads
    }


@app.route("/performance-prediction/<filename>")
def performance_prediction(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    try:
        analysis = analyze_advertisement(image_path)
        prediction = predict_ad_performance(analysis, limit=5)

        return render_template(
            "performance.html",
            filename=filename,
            analysis=analysis,
            prediction=prediction
        )

    except Exception as e:
        return f"Performance prediction error: {e}", 500


@app.route("/performance-prediction-test/<filename>")
def performance_prediction_test(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return {
            "status": "error",
            "message": "Advertisement image not found!"
        }, 404

    try:
        analysis = analyze_advertisement(image_path)
        prediction = predict_ad_performance(analysis, limit=5)

        return {
            "status": "success",
            "message": "Performance prediction completed successfully.",
            "uploaded_analysis": analysis,
            "prediction": prediction
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }, 500

# ==================================================
# AUDIENCE RECOMMENDATION PAGE
# ==================================================

@app.route("/audience-recommendation/<filename>")
def audience_recommendation(filename):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):

        return "Advertisement image not found!"

    try:

        # Analyze uploaded advertisement
        analysis = analyze_advertisement(
            image_path
        )

        # Generate audience recommendation
        recommendation = get_audience_recommendation(
            analysis,
            limit=5
        )

        return render_template(
            "audience.html",

            filename=filename,

            analysis=analysis,

            recommendation=recommendation
        )

    except Exception as e:

        return f"""
        <h2>Audience Recommendation Error</h2>
        <p>{str(e)}</p>
        """
@app.route("/competitor-intelligence/<filename>")
def competitor_intelligence(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    try:

        # Analyze uploaded advertisement
        analysis = analyze_advertisement(image_path)

        # Get competitor intelligence
        competitor_data = get_competitor_intelligence(
            analysis,
            limit=5
        )

        return render_template(
            "competitor.html",
            filename=filename,
            analysis=analysis,
            competitor=competitor_data
        )

    except Exception as e:

        return f"""
        <h2>Competitor Intelligence Error</h2>
        <p>{str(e)}</p>
        """
@app.route("/cost-estimator/<filename>")
def cost_estimator(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    try:

        analysis = analyze_advertisement(image_path)

        cost_data = get_cost_estimate(
            analysis,
            limit=5
        )

        return render_template(
            "cost_estimator.html",
            filename=filename,
            analysis=analysis,
            cost=cost_data
        )

    except Exception as e:

        return f"""
        <h2>Cost Estimator Error</h2>
        <p>{str(e)}</p>
        """
@app.route("/ad-fatigue/<filename>")
def ad_fatigue(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    try:

        analysis = analyze_advertisement(
            image_path
        )

        fatigue = get_ad_fatigue_analysis(
            analysis,
            limit=5
        )

        return render_template(
            "ad_fatigue.html",
            filename=filename,
            analysis=analysis,
            fatigue=fatigue
        )

    except Exception as e:

        return f"""
        <h2>Ad Fatigue Error</h2>
        <p>{str(e)}</p>
        """
@app.route("/final-recommendation/<filename>")
def final_recommendation(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    try:

        analysis = analyze_advertisement(
            image_path
        )

        recommendation = get_final_campaign_recommendation(
            analysis,
            limit=5
        )

        return render_template(
            "final_recommendation.html",
            filename=filename,
            analysis=analysis,
            recommendation=recommendation
        )

    except Exception as e:

        return f"""
        <h2>Final Recommendation Error</h2>
        <p>{str(e)}</p>
        """
@app.route("/download-report/<filename>")
def download_report(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(image_path):
        return "Advertisement image not found!"

    try:

        analysis = analyze_advertisement(
            image_path
        )

        recommendation = get_final_campaign_recommendation(
            analysis,
            limit=5
        )

        pdf_filename = generate_campaign_pdf(
            filename,
            analysis,
            recommendation
        )

        return send_from_directory(
            os.path.join(
                app.static_folder,
                "reports"
            ),
            pdf_filename,
            as_attachment=True
        )

    except Exception as e:

        return f"""
        <h2>PDF Report Error</h2>
        <p>{str(e)}</p>
        """
# ============================================================
# CAMPAIGN AI ASSISTANT
# ===========================================================
# ============================================================
# CAMPAIGN AI ASSISTANT
# ============================================================

@app.route("/chat/<filename>", methods=["GET"])
def chat_page(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "chatbot.html",
        filename=filename
    )


@app.route("/chat", methods=["POST"])
def chat_api():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "response": "Please login first."
        }), 401

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "response": "No message received."
            }), 400

        message = str(
            data.get("message", "")
        ).strip()

        filename = str(
            data.get("filename", "")
        ).strip()

        if not message:

            return jsonify({
                "success": False,
                "response": "Please type your question."
            }), 400

        if not filename:

            return jsonify({
                "success": False,
                "response": "Please select an advertisement first."
            }), 400


        # ----------------------------------------------------
        # Find uploaded advertisement
        # ----------------------------------------------------

        upload_folder = app.config["UPLOAD_FOLDER"]

        filepath = os.path.join(
            upload_folder,
            filename
        )

        if not os.path.exists(filepath):

            return jsonify({
                "success": False,
                "response":
                    "The selected advertisement was not found."
            }), 404


        # ----------------------------------------------------
        # Analyze advertisement
        # ----------------------------------------------------

        analysis = analyze_advertisement(filepath)

        if not analysis:

            return jsonify({
                "success": False,
                "response":
                    "Unable to analyze the advertisement."
            }), 500


        # Convert question to lowercase
        question = message.lower()


        # ====================================================
        # PERFORMANCE PREDICTION
        # ====================================================

        if any(word in question for word in [
            "performance",
            "predicted ctr",
            "ctr",
            "engagement",
            "conversion",
            "purchase",
            "click"
        ]):

            result = predict_ad_performance(
                analysis,
                limit=5
            )


            predicted_ctr = float(
                result.get(
                    "predicted_ctr",
                    0
                ) or 0
            )


            predicted_engagement = float(
                result.get(
                    "predicted_engagement_rate",
                    0
                ) or 0
            )


            click_purchase = float(
                result.get(
                    "predicted_click_to_purchase_rate",
                    0
                ) or 0
            )


            # ------------------------------------------------
            # Calculate purchases per 1000 impressions
            #
            # CTR = clicks / impressions
            # Purchase rate = purchases / clicks
            #
            # Therefore:
            #
            # purchases / 1000 impressions =
            # (CTR / 100) * (purchase rate / 100) * 1000
            # ------------------------------------------------

            estimated_purchases = (
                (predicted_ctr / 100)
                *
                (click_purchase / 100)
                *
                1000
            )


            response = f"""
            <b>📊 Performance Prediction</b><br><br>

            Based on historical similar advertisements:<br><br>

            <b>Predicted CTR:</b>
            {predicted_ctr:.2f}%<br>

            <b>Predicted Engagement Rate:</b>
            {predicted_engagement:.2f}%<br>

            <b>Predicted Click → Purchase Rate:</b>
            {click_purchase:.2f}%<br>

            <b>Estimated Purchases / 1,000 Impressions:</b>
            {estimated_purchases:.2f}<br><br>

            <small>
            This is a historical similarity-based estimate,
            not a live Meta prediction or guarantee.
            </small>
            """


        # ====================================================
        # AUDIENCE RECOMMENDATION
        # ====================================================

        elif any(word in question for word in [
            "audience",
            "target audience",
            "target customer",
            "age group",
            "gender",
            "interest",
            "interests"
        ]):

            result = get_audience_recommendation(
                analysis,
                limit=5
            )


            # ------------------------------------------------
            # Read possible keys from audience module
            # ------------------------------------------------

            age_group = result.get(
                "recommended_age_group"
            )

            if not age_group:
                age_group = result.get(
                    "age_group"
                )

            if not age_group:
                age_group = result.get(
                    "target_age_group"
                )


            gender = result.get(
                "recommended_gender"
            )

            if not gender:
                gender = result.get(
                    "gender"
                )

            if not gender:
                gender = result.get(
                    "target_gender"
                )


            interests = result.get(
                "recommended_interests"
            )

            if not interests:
                interests = result.get(
                    "interests"
                )

            if not interests:
                interests = result.get(
                    "target_interests"
                )


            platforms = result.get(
                "recommended_platforms"
            )

            if not platforms:
                platforms = result.get(
                    "platforms"
                )

            if not platforms:
                platforms = result.get(
                    "ad_platform"
                )


            # ------------------------------------------------
            # FALLBACK:
            # If audience result does not expose the fields,
            # derive them from the same similar advertisements.
            # ------------------------------------------------

            similar_ads = get_similar_ads(
                analysis,
                limit=5
            )


            # Age fallback
            if not age_group:

                age_values = []

                for ad in similar_ads:

                    value = ad.get(
                        "target_age_group"
                    )

                    if value:

                        age_values.append(
                            str(value)
                        )


                if age_values:

                    from collections import Counter

                    age_group = Counter(
                        age_values
                    ).most_common(1)[0][0]


            # Gender fallback
            if not gender:

                gender_values = []

                for ad in similar_ads:

                    value = ad.get(
                        "target_gender"
                    )

                    if value:

                        gender_values.append(
                            str(value)
                        )


                if gender_values:

                    from collections import Counter

                    gender = Counter(
                        gender_values
                    ).most_common(1)[0][0]


            # Interests fallback
            if not interests:

                interest_values = []

                for ad in similar_ads:

                    value = ad.get(
                        "target_interests"
                    )

                    if value:

                        if isinstance(value, list):

                            interest_values.extend(
                                value
                            )

                        else:

                            interest_values.append(
                                str(value)
                            )


                if interest_values:

                    cleaned_interests = []

                    for value in interest_values:

                        parts = str(value).replace(
                            ";",
                            ","
                        ).split(",")

                        for part in parts:

                            part = part.strip()

                            if part:
                                cleaned_interests.append(
                                    part
                                )


                    interests = list(
                        dict.fromkeys(
                            cleaned_interests
                        )
                    )


            # Platform fallback
            if not platforms:

                platform_values = []

                for ad in similar_ads:

                    value = ad.get(
                        "ad_platform"
                    )

                    if value:

                        platform_values.append(
                            str(value)
                        )


                if platform_values:

                    platforms = list(
                        dict.fromkeys(
                            platform_values
                        )
                    )


            # ------------------------------------------------
            # Final formatting
            # ------------------------------------------------

            if isinstance(interests, list):

                interests_text = ", ".join(
                    str(x)
                    for x in interests
                    if str(x).strip()
                )

            else:

                interests_text = str(
                    interests or ""
                )


            if isinstance(platforms, list):

                platforms_text = ", ".join(
                    str(x)
                    for x in platforms
                    if str(x).strip()
                )

            else:

                platforms_text = str(
                    platforms or ""
                )


            age_group = (
                str(age_group)
                if age_group
                else "Not available"
            )


            gender = (
                str(gender)
                if gender
                else "Not available"
            )


            interests_text = (
                interests_text
                if interests_text
                else "Not available"
            )


            platforms_text = (
                platforms_text
                if platforms_text
                else "Not available"
            )


            response = f"""
            <b>🎯 Audience Recommendation</b><br><br>

            <b>Recommended Age Group:</b>
            {age_group}<br>

            <b>Gender:</b>
            {gender}<br>

            <b>Interests:</b>
            {interests_text}<br>

            <b>Platforms:</b>
            {platforms_text}<br><br>

            <b>Reason:</b><br>

            Recommendation is based on the targeting
            patterns of historically similar advertisements
            and their observed engagement and purchase activity.
            """


        # ====================================================
        # BEST RELEASE TIME
        # ====================================================

        elif any(word in question for word in [
            "best time",
            "release time",
            "release",
            "when should",
            "timing",
            "schedule"
        ]):

            result = get_best_release_time(
                analysis,
                limit=5
            )


            best_day = result.get(
                "best_day",
                result.get(
                    "day",
                    "Not available"
                )
            )


            best_time = result.get(
                "best_time",
                result.get(
                    "time",
                    "Not available"
                )
            )


            release_score = float(
                result.get(
                    "release_score",
                    0
                ) or 0
            )


            events_analyzed = result.get(
                "events_analyzed",
                0
            )


            response = f"""
            <b>⏰ Best Release Time</b><br><br>

            <b>Recommended Day:</b>
            {best_day}<br>

            <b>Recommended Time:</b>
            {best_time}<br>

            <b>Release Score:</b>
            {release_score:.2f}<br>

            <b>Historical Events Analyzed:</b>
            {events_analyzed}<br><br>

            <small>
            This timing recommendation is derived from
            historical activity of similar advertisements.
            </small>
            """


        # ====================================================
        # SIMILAR ADS
        # ====================================================

        elif any(word in question for word in [
            "similar ads",
            "similar advertisement",
            "similar advertisements",
            "compare ads",
            "comparison"
        ]):

            similar_ads = get_similar_ads(
                analysis,
                limit=5
            )


            if not similar_ads:

                response = """
                <b>🔎 Similar Advertisements</b><br><br>

                No similar advertisements were found.
                """

            else:

                response = """
                <b>🔎 Similar Advertisements</b><br><br>

                Here are the most similar historical
                advertisements:<br><br>
                """


                for index, ad in enumerate(
                    similar_ads,
                    1
                ):

                    ad_id = ad.get(
                        "ad_id",
                        "N/A"
                    )

                    platform = ad.get(
                        "ad_platform",
                        "N/A"
                    )

                    ad_type = ad.get(
                        "ad_type",
                        "N/A"
                    )

                    age = ad.get(
                        "target_age_group",
                        "N/A"
                    )

                    gender = ad.get(
                        "target_gender",
                        "N/A"
                    )

                    similarity = float(
                        ad.get(
                            "similarity_score",
                            0
                        ) or 0
                    )


                    response += f"""
                    <b>{index}. Ad {ad_id}</b><br>

                    Platform:
                    {platform}<br>

                    Type:
                    {ad_type}<br>

                    Age Group:
                    {age}<br>

                    Gender:
                    {gender}<br>

                    Similarity:
                    {similarity:.2f}%<br><br>
                    """


        # ====================================================
        # TREND + REGIONAL
        # ====================================================

        elif any(word in question for word in [
            "trend",
            "regional",
            "region",
            "country",
            "location"
        ]):

            result = get_trend_regional_intelligence(
                analysis,
                limit=5
            )


            top_country = result.get(
                "top_country",
                "Not available"
            )


            top_location = result.get(
                "top_location",
                "Not available"
            )


            events_analyzed = result.get(
                "events_analyzed",
                0
            )


            response = f"""
            <b>🌍 Trend & Regional Intelligence</b><br><br>

            <b>Top Country:</b>
            {top_country}<br>

            <b>Top Location:</b>
            {top_location}<br>

            <b>Events Analyzed:</b>
            {events_analyzed}<br><br>

            <small>
            Regional insights are based on the available
            historical dataset and should not be treated
            as guaranteed future performance.
            </small>
            """


        # ====================================================
        # COMPETITOR INTELLIGENCE
        # ====================================================

        elif any(word in question for word in [
            "competitor",
            "competitors",
            "competition"
        ]):

            result = get_competitor_intelligence(
                analysis,
                limit=5
            )


            competitor_ads = result.get(
                "competitor_ads",
                result.get(
                    "ads",
                    []
                )
            )


            if not competitor_ads:

                response = """
                <b>🏆 Competitor Intelligence</b><br><br>

                No comparable competitor advertisements
                were found in the historical dataset.
                """

            else:

                response = """
                <b>🏆 Competitor Intelligence</b><br><br>

                Historically similar advertisements:<br><br>
                """


                for index, ad in enumerate(
                    competitor_ads,
                    1
                ):

                    ad_id = ad.get(
                        "ad_id",
                        "N/A"
                    )

                    platform = ad.get(
                        "ad_platform",
                        "N/A"
                    )

                    ad_type = ad.get(
                        "ad_type",
                        "N/A"
                    )

                    ctr = float(
                        ad.get(
                            "ctr",
                            0
                        ) or 0
                    )

                    purchases = ad.get(
                        "purchases",
                        0
                    )


                    response += f"""
                    <b>{index}. Ad {ad_id}</b><br>

                    Platform:
                    {platform}<br>

                    Type:
                    {ad_type}<br>

                    CTR:
                    {ctr:.2f}%<br>

                    Purchases:
                    {purchases}<br><br>
                    """


        # ====================================================
        # COST ESTIMATOR
        # ====================================================

        elif any(word in question for word in [
            "cost",
            "budget",
            "cpc",
            "cpm",
            "price",
            "spend"
        ]):

            result = get_cost_estimate(
                analysis,
                limit=5
            )


            estimated_cpm = float(
                result.get(
                    "estimated_cpm",
                    0
                ) or 0
            )


            estimated_cpc = float(
                result.get(
                    "estimated_cpc",
                    0
                ) or 0
            )


            estimated_budget = float(
                result.get(
                    "estimated_budget",
                    0
                ) or 0
            )


            target_impressions = result.get(
                "target_impressions",
                10000
            )


            response = f"""
            <b>💰 Campaign Cost Estimate</b><br><br>

            <b>Estimated CPM:</b>
            ₹{estimated_cpm:.2f}<br>

            <b>Estimated CPC:</b>
            ₹{estimated_cpc:.2f}<br>

            <b>Reference Budget:</b>
            ₹{estimated_budget:.2f}<br>

            <b>Target Impressions:</b>
            {target_impressions:,}<br><br>

            <small>
            This is a project-level historical estimate,
            not live Meta advertising pricing.
            </small>
            """


        # ====================================================
        # AD FATIGUE
        # ====================================================

        elif any(word in question for word in [
            "fatigue",
            "tired",
            "creative fatigue",
            "ad fatigue"
        ]):

            result = get_ad_fatigue_analysis(
                analysis,
                limit=5
            )


            fatigue_level = result.get(
                "fatigue_level",
                result.get(
                    "fatigue",
                    "Not available"
                )
            )


            fatigue_score = float(
                result.get(
                    "fatigue_score",
                    0
                ) or 0
            )


            ctr = float(
                result.get(
                    "ctr",
                    0
                ) or 0
            )


            engagement_rate = float(
                result.get(
                    "engagement_rate",
                    0
                ) or 0
            )


            purchase_rate = float(
                result.get(
                    "purchase_rate",
                    0
                ) or 0
            )


            recommendation = result.get(
                "recommendation",
                "Monitor performance and test alternative creatives."
            )


            response = f"""
            <b>🔄 Advertisement Fatigue</b><br><br>

            <b>Fatigue Level:</b>
            {fatigue_level}<br>

            <b>Fatigue Score:</b>
            {fatigue_score:.0f}/100<br>

            <b>CTR:</b>
            {ctr:.2f}%<br>

            <b>Engagement Rate:</b>
            {engagement_rate:.2f}%<br>

            <b>Purchase Rate:</b>
            {purchase_rate:.2f}%<br><br>

            <b>Recommendation:</b><br>

            {recommendation}
            """


        # ====================================================
        # FINAL CAMPAIGN RECOMMENDATION
        # ====================================================

        elif any(word in question for word in [
            "recommendation",
            "final recommendation",
            "campaign strategy",
            "strategy",
            "what should i do"
        ]):

            result = get_final_campaign_recommendation(
                analysis,
                limit=5
            )


            summary = result.get(
                "summary",
                "No summary available."
            )


            strategy = result.get(
                "campaign_strategy",
                "No campaign strategy available."
            )


            creative_action = result.get(
                "creative_action",
                "Monitor and optimize the creative."
            )


            monitoring = result.get(
                "monitoring_points",
                "CTR, engagement and conversion."
            )


            response = f"""
            <b>🚀 Final Campaign Recommendation</b><br><br>

            <b>Summary:</b><br>

            {summary}<br><br>

            <b>Campaign Strategy:</b><br>

            {strategy}<br><br>

            <b>Creative Action:</b><br>

            {creative_action}<br><br>

            <b>Monitoring Metrics:</b><br>

            {monitoring}
            """


        # ====================================================
        # GENERAL HELP
        # ====================================================

        else:

            response = """
            <b>🤖 Campaign AI Assistant</b><br><br>

            I can analyze your uploaded advertisement using
            the project's existing AI modules.<br><br>

            <b>You can ask:</b><br><br>

            📊 What is my predicted CTR?<br>

            🎯 Who is my target audience?<br>

            ⏰ What is the best release time?<br>

            🔎 Show similar advertisements<br>

            🌍 What region should I target?<br>

            🏆 Show competitor intelligence<br>

            💰 What is the estimated cost?<br>

            🔄 Is my ad suffering from fatigue?<br>

            🚀 Give me the final recommendation
            """


        # ====================================================
        # SEND RESPONSE
        # ====================================================

        return jsonify({
            "success": True,
            "response": response
        })


    except Exception as e:

        print(
            "[CHAT ERROR]",
            str(e)
        )

        return jsonify({
            "success": False,
            "response": """
            <b>⚠️ Assistant Error</b><br><br>

            Something went wrong while processing
            your question.<br><br>

            Please try again.
            """
        }), 500
# ==================================================
# LOGOUT
# ==================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )