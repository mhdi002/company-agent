"""Offline fixtures: fictional companies (reserved `.example` domains) and a field knowledge base.

All names, domains, and facts are invented for testing. They exercise search,
fetch/extract, field classification, research, SRLM programs, and synthetic
trace generation without network access. Nothing here describes a real company.
"""
from __future__ import annotations

COMPANIES: list[dict] = [
    dict(name="Nordwind Solar GmbH", domain="nordwind-solar.example", country="Germany", city="Hamburg",
         field="renewable energy", founded=2009, employees="120 employees",
         services=["utility-scale solar park development", "rooftop PV installation", "battery storage integration", "O&M services"],
         products=["SunGrid monitoring platform", "NW-Store 200 kWh battery cabinet"],
         clients="municipal utilities and logistics warehouses in northern Germany",
         about="Nordwind Solar develops, builds and operates solar parks and commercial rooftop systems. "
               "Since 2009 the company has connected more than 340 MW of photovoltaic capacity to the grid."),
    dict(name="Harbourline Logistics Ltd", domain="harbourline-logistics.example", country="United Kingdom", city="Felixstowe",
         field="logistics", founded=1998, employees="450 employees",
         services=["container drayage", "bonded warehousing", "customs brokerage", "last-mile delivery"],
         products=["PortTrack shipment visibility portal"],
         clients="importers of consumer electronics and furniture",
         about="Harbourline Logistics moves containers from the port of Felixstowe to distribution centres across England. "
               "The company runs 3 bonded warehouses with a combined 60,000 square metres."),
    dict(name="Maple Byte Software Inc.", domain="maplebyte.example", country="Canada", city="Toronto",
         field="software", founded=2014, employees="85 employees",
         services=["custom web application development", "cloud migration", "mobile app development", "QA automation"],
         products=["LedgerLite accounting SaaS", "ShiftPal staff scheduling app"],
         clients="small and medium-sized retailers and clinics",
         about="Maple Byte Software builds web and mobile software for growing businesses. "
               "Its SaaS products serve more than 2,000 small businesses across Canada."),
    dict(name="Outback AgriTech Pty Ltd", domain="outback-agritech.example", country="Australia", city="Toowoomba",
         field="agriculture", founded=2016, employees="40 employees",
         services=["soil moisture sensing", "precision irrigation design", "drone crop scouting"],
         products=["AquaProbe soil sensor", "FieldView irrigation controller"],
         clients="cotton and wheat growers in Queensland",
         about="Outback AgriTech helps broadacre farmers save water with sensors and precision irrigation. "
               "Growers using AquaProbe report water savings of up to 22 percent."),
    dict(name="Polderwerk Manufacturing B.V.", domain="polderwerk.example", country="Netherlands", city="Eindhoven",
         field="manufacturing", founded=1987, employees="300 employees",
         services=["precision CNC machining", "sheet metal fabrication", "contract assembly", "surface treatment"],
         products=["custom machine frames", "stainless steel enclosures"],
         clients="semiconductor equipment and food processing machine builders",
         about="Polderwerk is a contract manufacturer of precision metal parts and assemblies. "
               "The plant operates 45 CNC machines in a 20,000 square metre facility."),
    dict(name="Lion City MedTech Pte Ltd", domain="lioncity-medtech.example", country="Singapore", city="Singapore",
         field="healthcare", founded=2012, employees="60 employees",
         services=["remote patient monitoring", "clinical data integration", "telehealth platform operation"],
         products=["VitalLink wearable patch", "CareBridge clinician dashboard"],
         clients="private hospitals and elderly care homes",
         about="Lion City MedTech provides remote monitoring for patients with chronic conditions. "
               "VitalLink patches stream heart rate and oxygen saturation to clinicians in real time."),
    dict(name="Rheinland Freight Systems AG", domain="rheinland-freight.example", country="Germany", city="Duisburg",
         field="logistics", founded=2001, employees="900 employees",
         services=["rail freight forwarding", "intermodal terminal operation", "contract logistics"],
         products=["RailSlot booking system"],
         clients="steel and chemical producers in the Ruhr area",
         about="Rheinland Freight Systems operates intermodal terminals linking rail, river barges and trucks. "
               "Its Duisburg terminal handles about 400,000 TEU per year."),
    dict(name="Thames Ledger Analytics Ltd", domain="thamesledger.example", country="United Kingdom", city="London",
         field="finance", founded=2017, employees="35 employees",
         services=["credit risk modelling", "regulatory reporting automation", "financial data engineering"],
         products=["RiskLens scoring API"],
         clients="building societies and consumer lenders",
         about="Thames Ledger Analytics builds credit risk models and reporting pipelines for lenders. "
               "RiskLens scores more than one million loan applications each year."),
    dict(name="Prairie Build Co.", domain="prairiebuild.example", country="Canada", city="Calgary",
         field="construction", founded=1993, employees="210 employees",
         services=["commercial general contracting", "design-build", "prefabricated timber framing"],
         products=["PrairieFrame modular wall panels"],
         clients="school boards and commercial developers in Alberta",
         about="Prairie Build Co. is a general contractor delivering schools, offices and light industrial buildings. "
               "The company has completed more than 150 projects since 1993."),
    dict(name="Coral Coast Learning Pty Ltd", domain="coralcoast-learning.example", country="Australia", city="Cairns",
         field="education", founded=2019, employees="25 employees",
         services=["online vocational courses", "corporate training programs", "learning management system hosting"],
         products=["SkillReef learning platform"],
         clients="tourism operators and regional TAFE partners",
         about="Coral Coast Learning delivers online vocational training for the tourism and hospitality sector. "
               "More than 8,000 learners have completed a SkillReef course."),
    dict(name="Delta Fresh Retail B.V.", domain="deltafresh.example", country="Netherlands", city="Rotterdam",
         field="retail", founded=2005, employees="600 employees",
         services=["grocery retail", "online grocery delivery", "private label sourcing"],
         products=["DeltaFresh app"],
         clients="households in South Holland",
         about="Delta Fresh Retail runs 28 neighbourhood grocery stores and an online delivery service. "
               "Fresh produce is sourced from growers within 100 kilometres."),
    dict(name="Merlion Robotics Pte Ltd", domain="merlion-robotics.example", country="Singapore", city="Singapore",
         field="manufacturing", founded=2015, employees="70 employees",
         services=["industrial robot integration", "machine vision inspection", "automation consulting"],
         products=["VisionCell inspection station"],
         clients="electronics and pharmaceutical manufacturers",
         about="Merlion Robotics designs automated production cells with robots and machine vision. "
               "VisionCell detects surface defects at 120 parts per minute."),
]

# Field knowledge base used by the offline research provider. URLs are fictional (.example).
FIELD_KB: dict[str, list[dict]] = {
    "renewable energy": [
        {"url": "https://kb.example/renewable-energy/trends", "title": "Solar and storage trends",
         "text": "Co-locating battery storage with solar parks increases revenue from evening price peaks. "
                 "Grid connection queues are a major bottleneck for new solar projects. "
                 "Predictive maintenance using inverter data reduces downtime of PV plants. "
                 "Agrivoltaics combines crop production and solar generation on the same land."},
        {"url": "https://kb.example/renewable-energy/existing-work", "title": "Existing work in solar O&M",
         "text": "Operators already use SCADA monitoring for inverter alarms. "
                 "Drone thermography is widely used to find defective PV modules. "
                 "Few small developers have automated forecasting of solar output for energy trading."},
    ],
    "logistics": [
        {"url": "https://kb.example/logistics/trends", "title": "Logistics trends",
         "text": "Real-time shipment visibility is now expected by most importers. "
                 "Dynamic dock scheduling reduces truck waiting times at warehouses. "
                 "Electrification of drayage fleets requires depot charging planning. "
                 "Customs declarations are moving to fully digital submission systems."},
        {"url": "https://kb.example/logistics/existing-work", "title": "Existing work in freight digitisation",
         "text": "Many forwarders use transport management systems for route planning. "
                 "Container terminals publish slot booking APIs. "
                 "Demand forecasting for warehouse labour is still done in spreadsheets at many mid-size firms."},
    ],
    "software": [
        {"url": "https://kb.example/software/trends", "title": "Software services trends",
         "text": "Small businesses increasingly buy vertical SaaS instead of custom builds. "
                 "Automated testing pipelines shorten release cycles. "
                 "Usage analytics help SaaS vendors reduce customer churn. "
                 "Data privacy regulation requires audit logging in business software."},
        {"url": "https://kb.example/software/existing-work", "title": "Existing work in SMB SaaS",
         "text": "Accounting SaaS products commonly integrate with bank feeds. "
                 "Few scheduling apps forecast staffing demand from sales data."},
    ],
    "agriculture": [
        {"url": "https://kb.example/agriculture/trends", "title": "Precision agriculture trends",
         "text": "Variable-rate irrigation saves water in broadacre farming. "
                 "Satellite imagery provides weekly crop vigour maps at low cost. "
                 "Water allocation trading makes irrigation efficiency financially valuable."},
        {"url": "https://kb.example/agriculture/existing-work", "title": "Existing work in irrigation",
         "text": "Soil moisture probes are installed on many irrigated farms. "
                 "Integration of probe data with weather forecasts for irrigation scheduling remains limited."},
    ],
    "manufacturing": [
        {"url": "https://kb.example/manufacturing/trends", "title": "Manufacturing trends",
         "text": "Machine monitoring with OEE dashboards exposes hidden capacity. "
                 "Automated visual inspection reduces escaped defects. "
                 "Energy metering per machine supports carbon reporting requirements. "
                 "Quote automation shortens response times for contract manufacturers."},
        {"url": "https://kb.example/manufacturing/existing-work", "title": "Existing work in smart factories",
         "text": "CNC controllers expose spindle and alarm data over standard protocols. "
                 "Many job shops still schedule production on whiteboards."},
    ],
    "healthcare": [
        {"url": "https://kb.example/healthcare/trends", "title": "Digital health trends",
         "text": "Remote patient monitoring reduces hospital readmissions for chronic conditions. "
                 "Interoperability standards such as HL7 FHIR simplify clinical data exchange. "
                 "Alert fatigue is a key barrier to clinician adoption of monitoring tools."},
        {"url": "https://kb.example/healthcare/existing-work", "title": "Existing work in remote monitoring",
         "text": "Wearable patches already stream vital signs to dashboards. "
                 "Few systems prioritise alerts with validated risk scores."},
    ],
    "finance": [
        {"url": "https://kb.example/finance/trends", "title": "Lending technology trends",
         "text": "Open banking data improves affordability assessments. "
                 "Model risk management rules require explainable credit models. "
                 "Regulatory reporting is shifting to granular data submissions."},
    ],
    "construction": [
        {"url": "https://kb.example/construction/trends", "title": "Construction trends",
         "text": "Prefabrication shortens on-site schedules and reduces waste. "
                 "Digital site diaries and BIM models improve coordination. "
                 "Embodied carbon reporting is increasingly required by public clients."},
    ],
    "education": [
        {"url": "https://kb.example/education/trends", "title": "Vocational learning trends",
         "text": "Micro-credentials let workers upskill in short modules. "
                 "Learning analytics identify learners at risk of dropping out. "
                 "Mobile-first course design improves completion for shift workers."},
    ],
    "retail": [
        {"url": "https://kb.example/retail/trends", "title": "Grocery retail trends",
         "text": "Demand forecasting for fresh produce reduces food waste. "
                 "Online grocery orders require efficient picking in stores. "
                 "Dynamic markdowns of near-expiry items recover margin."},
    ],
}

PAGE_TEMPLATES = {
    "home": "{name} — {about}",
    "about": "About us. {name} was founded in {founded} in {city}, {country}. {about} We employ {employees}. "
             "Our customers include {clients}.",
    "services": "Our services: {services_text}. We work with {clients}.",
    "products": "Products: {products_text}.",
}


def company_pages(c: dict) -> dict[str, str]:
    """Return {url: html} for a fictional company website, with boilerplate noise."""
    services_text = "; ".join(c["services"])
    products_text = "; ".join(c["products"])
    fmt = dict(c, services_text=services_text, products_text=products_text)
    nav = "<nav><a href='/'>Home</a> | <a href='/about'>About</a> | <a href='/services'>Services</a> | <a href='/products'>Products</a> | <a href='/contact'>Contact</a></nav>"
    cookie = "<div class='cookie-banner'>We use cookies to improve your experience. Accept all cookies.</div>"
    footer = f"<footer>© {c['founded']}–2026 {c['name']}. All rights reserved. Privacy policy | Imprint</footer>"
    pages = {}
    for key, tpl in PAGE_TEMPLATES.items():
        path = "/" if key == "home" else f"/{key}"
        body = tpl.format(**fmt)
        items = ""
        if key == "services":
            items = "<ul>" + "".join(f"<li>{s}</li>" for s in c["services"]) + "</ul>"
        if key == "products":
            items = "<ul>" + "".join(f"<li>{p}</li>" for p in c["products"]) + "</ul>"
        html = (f"<!doctype html><html lang='en'><head><title>{c['name']} | {key.title()}</title>"
                f"<meta name='description' content='{c['about'][:150]}'>"
                f"<script>var tracking = 'x';</script><style>body{{font:14px sans-serif}}</style></head>"
                f"<body>{cookie}<header>{nav}</header><main><h1>{c['name']}</h1><p>{body}</p>{items}</main>{footer}</body></html>")
        pages[f"https://{c['domain']}{path}"] = html
    pages[f"https://{c['domain']}/contact"] = (
        f"<html><body>{nav}<main><h1>Contact</h1><p>{c['name']}, {c['city']}, {c['country']}.</p>"
        f"<p>General enquiries: info@{c['domain']}</p></main>{footer}</body></html>")
    pages[f"https://{c['domain']}/robots.txt"] = "User-agent: *\nDisallow: /admin\n"
    return pages


def all_pages() -> dict[str, str]:
    out: dict[str, str] = {}
    for c in COMPANIES:
        out.update(company_pages(c))
    return out
