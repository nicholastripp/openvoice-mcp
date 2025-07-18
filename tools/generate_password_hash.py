#!/usr/bin/env python3
"""
Generate bcrypt password hash for web UI authentication
"""
import sys
import getpass

try:
    import bcrypt
except ImportError:
    print("Error: bcrypt module not found. Please install requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def generate_hash(password: str) -> str:
    """Generate bcrypt hash for password"""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def main():
    print("Generate password hash for HA Voice Assistant Web UI")
    print("-" * 50)
    
    # Get password securely
    password = getpass.getpass("Enter password: ")
    confirm = getpass.getpass("Confirm password: ")
    
    if password != confirm:
        print("Error: Passwords don't match!")
        sys.exit(1)
    
    if not password:
        print("Error: Password cannot be empty!")
        sys.exit(1)
    
    # Generate hash
    password_hash = generate_hash(password)
    
    print("\nGenerated password hash:")
    print("-" * 50)
    print(password_hash)
    print("-" * 50)
    print("\nTo use this hash, update your config/config.yaml:")
    print("  web_ui:")
    print("    auth:")
    print(f'      password_hash: "{password_hash}"')


if __name__ == "__main__":
    main()