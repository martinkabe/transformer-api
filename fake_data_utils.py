# fake_value_functions.py

import random
from faker import Faker

fake = Faker()

FAKE_VALUE_FUNCTIONS = {
    # Bank
    "account_number": lambda: fake.iban(),
    "balance": lambda: round(fake.pyfloat(left_digits=5, right_digits=2), 2),
    "currency": lambda: fake.currency_code(),

    # School
    "student_id": lambda: fake.uuid4(),
    "grade": lambda: random.choice(["A", "B", "C", "D", "F"]),
    "email": lambda: fake.email(),

    # Hospital
    "patient_id": lambda: fake.uuid4(),
    "diagnosis": lambda: random.choice(["flu", "allergy", "injury"]),
    "doctor_name": lambda: fake.name(),

    # Company
    "employee_id": lambda: fake.uuid4(),
    "position": lambda: random.choice(["manager", "developer", "analyst"]),
    "salary": lambda: round(fake.pyfloat(left_digits=5, right_digits=2), 2),

    # Sport
    "member_id": lambda: fake.uuid4(),
    "sport": lambda: random.choice(["football", "tennis", "basketball"]),
    "membership_status": lambda: random.choice(["active", "inactive"]),

    # Ecommerce
    "product_id": lambda: fake.uuid4(),
    "product_name": lambda: fake.word().capitalize(),
    "price": lambda: round(fake.pyfloat(left_digits=3, right_digits=2), 2),
    "category": lambda: random.choice(["electronics", "clothing", "books", "furniture"]),

    # Travel
    "trip_id": lambda: fake.uuid4(),
    "destination": lambda: fake.city(),
    "departure_date": lambda: fake.date_this_year(),
    "return_date": lambda: fake.date_this_year(),

    # Restaurant
    "menu_item": lambda: fake.word().capitalize(),
    "ingredients": lambda: ", ".join(fake.words(3)),

    # Weather
    "location": lambda: fake.city(),
    "temperature": lambda: round(fake.pyfloat(left_digits=2, right_digits=1), 1),
    "humidity": lambda: random.randint(10, 100),
    "forecast": lambda: random.choice(["sunny", "rainy", "cloudy", "stormy", "snowy"]),

    # Transport
    "vehicle_id": lambda: fake.uuid4(),
    "type": lambda: random.choice(["bus", "train", "truck", "car"]),
    "capacity": lambda: random.randint(2, 100),
    "route": lambda: fake.bothify(text="Route-##??")
}

def generate_fake_data(columns, n=5):
    rows = []
    for _ in range(n):
        row = [FAKE_VALUE_FUNCTIONS.get(col, lambda: fake.word())() for col in columns]
        rows.append(row)
    return {"columns": columns, "rows": rows}