"""Synthetic data generator for Business Entity Resolution Challenge.
Generates realistic training and test data covering US, India, and France with realistic noise.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

# Base business templates across countries
TRAIN_TEMPLATES = [
    # US Businesses
    ("Acme Industrial Supply", "1042 Market Street, Suite 400, San Francisco, CA 94103", "US"),
    ("Pinnacle Global Logistics", "450 Lexington Avenue, Floor 12, New York, NY 10017", "US"),
    ("Apex Health Solutions", "789 Peachtree St NE, Atlanta, GA 30308", "US"),
    ("Summit Financial Partners", "200 South Wacker Dr, Chicago, IL 60606", "US"),
    ("Blue Horizon Media", "1200 Sunset Blvd, Los Angeles, CA 90028", "US"),
    ("Redwood Cloud Systems", "3000 Sand Hill Road, Building 4, Menlo Park, CA 94025", "US"),
    ("Vanguard Biotech Labs", "500 Kendall Street, Cambridge, MA 02142", "US"),
    ("Beacon Energy Services", "1000 Louisiana St, Suite 2200, Houston, TX 77002", "US"),
    ("Silverline Hardware Tools", "2500 Industrial Pkwy, Cleveland, OH 44135", "US"),
    ("Crestview Property Holdings", "150 2nd Avenue North, St. Petersburg, FL 33701", "US"),
    
    # India Businesses
    ("Tata Consultancy Infotech", "Plot 42, Electronics City Phase 1, Hosur Road, Bangalore, Karnataka 560100", "India"),
    ("Reliance Retail Enterprises", "Maker Chambers IV, Nariman Point, Mumbai, Maharashtra 400021", "India"),
    ("Infosys Digital Solutions", "Electronics City, Hosur Road, Bengaluru 560100", "India"),
    ("Hindustan Commercial Logistics", "12 Netaji Subhash Road, Fairley Place, BBD Bagh, Kolkata, WB 700001", "India"),
    ("Larsen Heavy Engineering", "L&T House, Ballard Estate, Mumbai, Maharashtra 400001", "India"),
    ("Godrej Agrovet Enterprises", "Pirojshanagar, Eastern Express Highway, Vikhroli East, Mumbai 400079", "India"),
    ("Apollo Healthcare Clinics", "19 Bishop Gardens, Raja Annamalaipuram, Chennai, Tamil Nadu 600028", "India"),
    ("Mahindra Automotive Spares", "Gateway Building, Apollo Bunder, Colaba, Mumbai 400001", "India"),
    ("Bajaj Consumer Care", "221 CST Road, Kalina, Santacruz East, Mumbai 400098", "India"),
    ("Wipro Strategic Technologies", "Doddakannelli, Sarjapur Road, Bangalore 560035", "India"),
    
    # Singletons (Will have NO matches in S2 or S3)
    ("Evergreen Organic Foods", "880 Broadway, Boulder, CO 80302", "US"),
    ("Zenith Robotics Labs", "15 Tech Park Drive, Austin, TX 78701", "US"),
    ("Shree Balaji Textiles", "Ring Road, Surat, Gujarat 395002", "India"),
    ("Kaveri Electrical Works", "14 SP Road, Bengaluru, Karnataka 560002", "India"),
]

TEST_TEMPLATES = [
    # US
    ("Paramount Software Systems", "600 Congress Ave, Austin, TX 78701", "US"),
    ("Keystone Capital Management", "100 Federal Street, Boston, MA 02110", "US"),
    ("Starlight Retail Stores", "750 7th Avenue, New York, NY 10019", "US"),
    ("Cascade Mountain Brewing", "1200 NW 23rd Ave, Portland, OR 97210", "US"),
    
    # India
    ("Zomato Online Deliveries", "Ground Floor, 12A 94 Meghdoot, Nehru Place, New Delhi 110019", "India"),
    ("Swiggy Bundl Technologies", "Maruthi Chambers, Survey No 17/9B, Roopena Agrahara, Bangalore 560068", "India"),
    ("Bharat Petroleum Refinery", "Bharat Bhavan, 4 and 6 Currimbhoy Road, Ballard Estate, Mumbai 400001", "India"),
    ("Dr Reddys Laboratories", "8-2-337 Road No 3, Banjara Hills, Hyderabad, Telangana 500034", "India"),

    # France (Appears only in test as specified in problem description!)
    ("LVMH Moet Hennessy", "22 Avenue Montaigne, 75008 Paris", "France"),
    ("Danone Produits Frais", "17 Boulevard Haussmann, 75009 Paris", "France"),
    ("Schneider Electric France", "35 Rue Joseph Monier, 92500 Rueil-Malmaison", "France"),
    ("TotalEnergies Renouvelables", "2 Place Jean Millier, La Defense 6, 92400 Courbevoie", "France"),
    ("Carrefour Hypermarchés", "93 Avenue de Paris, 91300 Massy", "France"),

    # Singletons in test set
    ("Mont Blanc Alpine Ski", "45 Rue du Mont, 74400 Chamonix", "France"),
    ("Kashmir Artisans Cooperative", "Residency Road, Srinagar, J&K 190001", "India"),
    ("Mojave Solar Innovations", "700 Desert View Dr, Barstow, CA 92311", "US"),
]

def corrupt_name(name: str, country: str, source: int) -> str:
    """Simulate realistic noise in business name."""
    s = name
    if source == 2:
        # Legal suffix variation & punctuation
        s = s.replace("Consultancy", "Consulting").replace("Industrial", "Ind.")
        s = s.replace("Solutions", "Solns").replace("Enterprises", "Ent.")
        s = s.replace("and", "&")
        if country == "India":
            s += " Pvt. Ltd."
        elif country == "US":
            s += ", Inc."
        elif country == "France":
            s += " SAS"
        # Occasional minor typo
        if random.random() < 0.3 and len(s) > 8:
            idx = random.randint(2, len(s) - 3)
            s = s[:idx] + s[idx + 1] + s[idx] + s[idx + 2 :]
    elif source == 3:
        # Word transposition / abbreviation
        words = s.split()
        if len(words) >= 3 and random.random() < 0.5:
            # Word swap
            s = f"{words[-1]} {words[0]} {' '.join(words[1:-1])}"
        if country == "India":
            s += " Private Limited"
        elif country == "US":
            s += " Corporation"
        elif country == "France":
            s += " SARL"
    return s

def corrupt_address(addr: str, country: str, source: int) -> str:
    """Simulate realistic address variations, missing PIN, landmarks."""
    s = addr
    if source == 2:
        # Abbreviate road types
        s = s.replace("Street", "St.").replace("Avenue", "Ave").replace("Road", "Rd")
        s = s.replace("Boulevard", "Blvd").replace("Floor", "Fl").replace("Suite", "Ste")
        if country == "India" and random.random() < 0.6:
            s = "Near SBI Bank ATM, " + s
        elif country == "US" and random.random() < 0.5:
            s = s.split(",")[0] + ", " + s.split(",")[-1]  # missing middle
    elif source == 3:
        # Drop PIN / Zip or reorder
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            # reverse order of street and building
            s = f"{parts[1]}, {parts[0]}, {', '.join(parts[2:])}"
        if country == "India":
            # Strip 6-digit PIN code
            import re
            s = re.sub(r"\b\d{6}\b", "", s).strip()
            s += " Opp Metro Pillar 42"
    return s

def generate_synthetic_dataset(base_dir: Path):
    """Generate train and test datasets matching the exact challenge schema and file layout."""
    base_dir.mkdir(parents=True, exist_ok=True)
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    for mode, templates, target_dir in [
        ("train", TRAIN_TEMPLATES, train_dir),
        ("test", TEST_TEMPLATES, test_dir),
    ]:
        s1_rows = []
        s2_rows = []
        s3_rows = []
        gt_rows = []

        s2_counter = 1
        s3_counter = 1

        for i, (name, addr, country) in enumerate(templates, start=1):
            s1_id = f"S1-{i:05d}"
            s1_rows.append(f"{s1_id}\t{name}\t{addr}\t{country}\n")

            matched_ids = []

            # Check if this is a singleton
            is_singleton = "Singletons" in name or i > len(templates) - 4

            if not is_singleton:
                # Add to S2 (90% chance)
                if random.random() < 0.90:
                    s2_id = f"S2-{s2_counter:05d}"
                    s2_counter += 1
                    s2_name = corrupt_name(name, country, source=2)
                    s2_addr = corrupt_address(addr, country, source=2)
                    s2_rows.append(f"{s2_id}\t{s2_name}\t{s2_addr}\t{country}\n")
                    matched_ids.append(s2_id)

                # Add to S3 (85% chance)
                if random.random() < 0.85:
                    s3_id = f"S3-{s3_counter:05d}"
                    s3_counter += 1
                    s3_name = corrupt_name(name, country, source=3)
                    s3_addr = corrupt_address(addr, country, source=3)
                    s3_rows.append(f"{s3_id}\t{s3_name}\t{s3_addr}\t{country}\n")
                    matched_ids.append(s3_id)

            if mode == "train":
                gt_str = ",".join(matched_ids)
                gt_rows.append(f"{s1_id}\t{gt_str}\n")

        # Write files
        prefix = f"{mode}_"
        with open(target_dir / f"{prefix}source1.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.writelines(s1_rows)

        with open(target_dir / f"{prefix}source2.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.writelines(s2_rows)

        with open(target_dir / f"{prefix}source3.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.writelines(s3_rows)

        if mode == "train":
            with open(target_dir / f"{prefix}ground_truth.tsv", "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                f.writelines(gt_rows)

    print(f"Generated synthetic dataset successfully in {base_dir}")

if __name__ == "__main__":
    generate_synthetic_dataset(Path("dataset"))
