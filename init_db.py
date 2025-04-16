import database

# Initialize the database
print("Initializing database...")
success = database.initialize()

if success:
    print("Database initialized successfully!")
else:
    print("Failed to initialize database.")