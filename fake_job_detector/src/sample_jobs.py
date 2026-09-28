"""
sample_jobs.py
--------------
DEMO / TEST examples written by hand for this project.
They are NOT verified real-world job advertisements. "expected" is only what a
human would guess, so we can check that the app behaves sensibly.
"""

SAMPLE_JOBS = [
    {
        "name": "Demo 1 - Likely genuine (software developer)",
        "expected": "likely genuine",
        "title": "Junior Python Developer",
        "company_profile": (
            "Brightwave Software is a product company founded in 2011 with 120 employees "
            "across Chennai and Bengaluru. We build inventory and billing software for "
            "small retailers and have served over 3,000 customers."
        ),
        "description": (
            "We are looking for a Junior Python Developer to join our backend team. You will "
            "write and test REST APIs, fix bugs, review code with senior engineers, and take "
            "part in weekly sprint planning. You will work with Python, Django and PostgreSQL "
            "and learn our deployment process. The role is based in our Chennai office with "
            "two remote days per week."
        ),
        "requirements": (
            "B.E./B.Tech in Computer Science or related field. Good knowledge of Python and "
            "SQL. Understanding of data structures and Git. Good communication skills."
        ),
        "benefits": "Health insurance, paid leave, learning budget, annual performance bonus.",
        "location": "IN, TN, Chennai",
        "salary_range": "400000-600000",
        "employment_type": "Full-time",
        "education": "Bachelor's Degree",
        "experience": "Entry level",
        "industry": "Information Technology and Services",
        "function": "Information Technology",
        "telecommuting": False, "has_logo": True, "has_questions": True,
    },
    {
        "name": "Demo 2 - Likely genuine (staff nurse)",
        "expected": "likely genuine",
        "title": "Registered Nurse - Medical Ward",
        "company_profile": (
            "Greenfield Community Hospital is a 250-bed non-profit hospital that has served "
            "the region for more than 40 years, with accredited departments in surgery, "
            "paediatrics and emergency care."
        ),
        "description": (
            "Greenfield Community Hospital is hiring a Registered Nurse for the medical ward. "
            "Responsibilities include administering medication, monitoring patients, "
            "maintaining records and working with doctors and the care team. Shifts are "
            "rotating and include some weekends. Full orientation and mentoring are provided."
        ),
        "requirements": (
            "Valid nursing licence, BSc Nursing or GNM, at least one year of clinical "
            "experience, basic life support certification."
        ),
        "benefits": "Medical cover, shift allowance, paid holidays, continuing education support.",
        "location": "US, OH, Columbus",
        "salary_range": "",
        "employment_type": "Full-time",
        "education": "Bachelor's Degree",
        "experience": "Mid-Senior level",
        "industry": "Hospital & Health Care",
        "function": "Health Care Provider",
        "telecommuting": False, "has_logo": True, "has_questions": False,
    },
    {
        "name": "Demo 3 - Suspicious (work-from-home data entry)",
        "expected": "suspicious",
        "title": "EARN $5000 WEEKLY - DATA ENTRY WORK FROM HOME!!!",
        "company_profile": "",
        "description": (
            "URGENT HIRING!!! No experience needed. Earn up to $5000 per week from home, "
            "guaranteed income, unlimited income. Just type simple data on your phone or "
            "laptop. Limited slots, apply now and start immediately. Pay a small registration "
            "fee of $50 to receive your training kit. Send your personal details and bank "
            "account number to hr.recruit2024@gmail.com or message us on WhatsApp +1 555 013 4477."
        ),
        "requirements": "",
        "benefits": "Get paid daily. Be your own boss.",
        "location": "",
        "salary_range": "5000-20000",
        "employment_type": "Other",
        "education": "",
        "experience": "",
        "industry": "",
        "function": "",
        "telecommuting": True, "has_logo": False, "has_questions": False,
    },
    {
        "name": "Demo 4 - Suspicious (overseas 'payment processor')",
        "expected": "suspicious",
        "title": "Payment Processing Agent (Immediate Start)",
        "company_profile": "",
        "description": (
            "International company needs payment agents to receive client payments into "
            "their own bank account and forward the money by wire transfer or Western Union, "
            "keeping a 10 percent commission. Easy money, no interview, start today. "
            "Contact the manager on Telegram."
        ),
        "requirements": "Must have an active bank account.",
        "benefits": "",
        "location": "",
        "salary_range": "",
        "employment_type": "Part-time",
        "education": "",
        "experience": "",
        "industry": "",
        "function": "",
        "telecommuting": True, "has_logo": False, "has_questions": False,
    },
    {
        "name": "Demo 5 - Suspicious (vague airline job with fee)",
        "expected": "suspicious",
        "title": "Airline Ground Staff - Apply Now",
        "company_profile": "",
        "description": (
            "Big airline vacancies!!! Salary 90000 monthly, free accommodation. Selected "
            "candidates must pay a training fee and security deposit before joining. "
            "Hurry, last chance, limited seats. Email your documents to jobs.airline.hr@yahoo.com."
        ),
        "requirements": "",
        "benefits": "Free accommodation, free tickets",
        "location": "",
        "salary_range": "90000",
        "employment_type": "Full-time",
        "education": "",
        "experience": "",
        "industry": "",
        "function": "",
        "telecommuting": False, "has_logo": False, "has_questions": False,
    },
]
