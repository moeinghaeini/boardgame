import zipfile
import os
import pandas as pd
import requests
import time
import math
import xml.etree.ElementTree as ET

# File Paths
zip_file_path = './data/boardgames_ranks_2024-08-24.zip'
extract_dir = './data/boardgames_rank_2024-08-24/'
csv_output_path = './data/boardgames_comments.csv'

# Function to extract the zip file
def extract_zip(zip_file_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted files to {extract_dir}")

#Downloading comments
def download_game_comments(game_id):
    comments = []
    base_url = f'https://api.geekdo.com/xmlapi2/thing?type=boardgame&id={game_id}&comments=1'

    print(f'Downloading comments for game with id {game_id}')

    response = requests.get(base_url)
    root = ET.fromstring(response.content)

    number_of_comments = int(root[0].find('comments').attrib['totalitems'])
    number_of_pages = math.ceil(number_of_comments / 100)
    time.sleep(1)

    # Downloading comments from all pages
    for page in range(1, number_of_pages + 1):
        url = f'{base_url}&page={page}'
        page_comments = download_page_comments(url, game_id)
        comments.extend(page_comments)

        #used in order to verify the no. of comments downloaded
        print(f"{len(comments)}/{number_of_comments} comments downloaded.")
        time.sleep(1.5)

    return pd.DataFrame(comments)


def download_page_comments(url, game_id):
    while True:
        response = requests.get(url)
        root = ET.fromstring(response.content)

        # Extracting comments from the current page
        page_comments = []
        for comment in root.iter('comment'):
            comment_data = comment.attrib
            comment_data['boardgame_id'] = game_id
            page_comments.append(comment_data)

        if page_comments:
            return page_comments
        else:
            print('Repeating page download since no comments were received')
            time.sleep(1)

# Main function
def main():
    # Unzipping the rankings file
    extract_zip(zip_file_path, extract_dir)

    # Loading the top 10 board games from the CSV
    csv_file_path = os.path.join(extract_dir, 'boardgames_ranks.csv')
    df_bgs = pd.read_csv(csv_file_path)
    df_bgs_10 = df_bgs[:10]
    print("Top 10 games extracted from the rankings CSV.")

    # Downloading comments for the top 10 games
    if not os.path.isfile(csv_output_path):
        df_comments = pd.DataFrame()

        for i, boardgame in df_bgs_10.iterrows():
            game_id = boardgame['id']
            df_game_comments = download_game_comments(game_id)
            df_comments = pd.concat([df_comments, df_game_comments])

        df_comments.to_csv(csv_output_path)
        print(f"Comments saved to {csv_output_path}")
    else:
        df_comments = pd.read_csv(csv_output_path, index_col=0)
        print(f"Comments loaded from {csv_output_path}")

if __name__ == "__main__":
    main()



