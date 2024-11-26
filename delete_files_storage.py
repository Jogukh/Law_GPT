import openai

# Initialize the client
client = openai.OpenAI(api_key='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')

# List files in your OpenAI storage
response = client.files.list()

# Initialize a counter for the total number of files
file_count = 0

# List to store file IDs for deletion
file_ids = []

# Iterate over the files and count them
for file in response:
    file_count += 1
    file_ids.append(file.id)
    print(f"ID: {file.id}, Filename: {file.filename}, Purpose: {file.purpose}")

# Print the total number of files
print(f"Total number of files: {file_count}")

# Ask user if they want to delete all files
delete_all = input("Would you like to delete all files? (Y/N): ").strip().upper()

if delete_all == 'Y':
    # Iterate over file IDs and delete each file
    for file_id in file_ids:
        client.files.delete(file_id)
        print(f"Deleted file with ID: {file_id}")
    print("All files have been deleted.")
else:
    print("No files were deleted.")