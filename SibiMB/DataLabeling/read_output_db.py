import sqlite3

# Connect to the output database
output_db = sqlite3.connect("output.db")


# Read the data from the output database
cursor = output_db.cursor()
cursor.execute("SELECT file_name, label FROM labels")
results = cursor.fetchall()

# Print the results
for file_name, label in results:
    print(f"File: {file_name} - Label: {label}")

# Close the connection
output_db.close()
