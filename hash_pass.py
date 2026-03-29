import streamlit_authenticator as stauth

# List of passwords to hash
passwords = ["password123", "password123"]

# Correct usage of Hasher in v0.4.2
hasher = stauth.Hasher(passwords)  # Initialize first
hashed_passwords = hasher.generate()  # Generate hashes

# Print hashed passwords
print(hashed_passwords)
