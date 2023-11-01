import sqlite3

# Connect to the input database (or create it if it doesn't exist)
input_db = sqlite3.connect("input.db")


# Create a table for storing the text files
input_db.execute("CREATE TABLE IF NOT EXISTS text_files (file_name TEXT, content TEXT)")


# Insert the text files into the database
data = [
    ("file1.txt", """Once upon a time, in a small village nestled in the mountains, there lived a young girl named Aria. Aria had a great passion for music and spent most of her days singing and playing her guitar. She dreamed of becoming a famous musician and sharing her love of music with the world.

One day, a music producer named David came to the village looking for fresh talent to sign to his record label. Aria was ecstatic when she heard the news and quickly got ready for her audition. She practiced for hours and poured her heart and soul into every note.

At the audition, Aria nervously took the stage and began to sing. Her voice was pure and powerful, and she captivated the room with her talent. David was blown away by her performance and offered her a record deal on the spot.

Over the next few months, Aria worked tirelessly in the studio, perfecting her songs and honing her craft. She toured the country, playing at sold-out venues and gaining a massive following of fans.

Despite her success, Aria never forgot her roots. She visited her hometown often and spent time with her family and friends. She even organized a music festival in the village, showcasing local talent and giving back to the community that had supported her from the beginning.

Years went by, and Aria continued to make music that touched people's hearts. She never lost her love of singing and playing guitar, and her music continued to inspire people all around the world. Aria became a true legend in the music industry, but to those who knew her best, she was still the same down-to-earth girl from the mountains.

And so, Aria's story became an inspiration to all those who dared to follow their dreams and pursue their passions. Her legacy lived on long after she was gone, and her music continued to bring joy to people's lives for generations to come."""),
    ("file2.txt", """Once upon a time, in a small fishing village by the sea, there lived a young boy named Kai. Kai loved nothing more than exploring the coastline and discovering new treasures washed up by the tide.

One day, while playing by the shore, Kai stumbled upon a mysterious bottle. Inside the bottle was a letter, written in a language he couldn't understand. But what caught his attention was the map that was enclosed with the letter, showing the location of a treasure hidden on a nearby island.

Kai knew he had to find the treasure, so he set off on a quest to decipher the language of the letter and follow the map. He spent countless hours studying ancient texts and learning the language, determined to uncover the secrets of the map.

Finally, after months of hard work, Kai was ready to set sail to the island. He packed his bag with supplies and set off on his journey, braving treacherous seas and battling fierce storms along the way.

When he arrived at the island, he used the map to guide him to the location of the treasure. It was buried deep in a cave, guarded by traps and obstacles designed to keep intruders out. But Kai was determined and used all of his skills to navigate the cave and retrieve the treasure.

As he emerged from the cave, Kai was amazed by what he saw. The treasure was not made of gold or jewels, but rather a collection of ancient artifacts and relics. They were priceless, each with its own story and history.

Kai knew he had uncovered something truly special, and he devoted his life to studying and preserving the artifacts. He became an expert in archaeology, traveling the world to uncover other lost treasures and learning about the civilizations that had come before.

Years went by, and Kai became known as one of the world's foremost archaeologists. His discoveries had shed light on ancient cultures and brought new knowledge to the world. And all because of the courage and determination he showed in pursuing his dream."""),
    # Add more entries as needed
]


# Insert the data into the text_files table
for file_name, content in data:
    input_db.execute("INSERT INTO text_files (file_name, content) VALUES (?, ?)", (file_name, content))

# Commit the changes and close the connection
input_db.commit()
input_db.close()
