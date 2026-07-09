import random

users = []

for i in range(1, 11):
    user = {
        "User_ID": f"U{i}",
        "Daily_App_Opens": random.randint(1, 25),
        "Avg_Session_Duration": round(random.uniform(2, 45), 2)  # minutes
    }
    users.append(user)

print(users)