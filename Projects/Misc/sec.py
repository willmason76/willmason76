import requests
import os

cik = '0000789570'.lstrip('0')      #MGM
base_url = f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'

headers = {'User-Agent': 'Will Mason (masonw1@unlv.nevada.edu)'}

response = requests.get(base_url, headers = headers)
data = response.json()

forms = data['filings']['recent']['form']
accessions = data['filings']['recent']['accessionNumber']
filing_dates = data['filings']['recent']['filingDate']

# Set how many 10-Ks you want to download
num_to_download = 10
downloaded = 0

for form, accession, date in zip(forms, accessions, filing_dates):
    if form == '10-K':
        formatted_cik = cik.zfill(10)
        clean_accession = accession.replace('-', '')
        index_url = f"https://www.sec.gov/Archives/edgar/data/{formatted_cik}/{clean_accession}/index.json"

        print(f"\nFetching filing index: {index_url}")
        index_response = requests.get(index_url, headers=headers)
        index_data = index_response.json()

        doc_info = None
        for doc in index_data['directory']['item']:
            name = doc['name']
            if name.endswith('.htm') or name.endswith('.html'):
                if '10-k' in name.lower():
                    doc_info = name
                    break
        if not doc_info:
            for doc in index_data['directory']['item']:
                if doc['name'].endswith('.htm'):
                    doc_info = doc['name']
                    break

        if doc_info:
            filing_txt_url = f"https://www.sec.gov/Archives/edgar/data/{formatted_cik}/{clean_accession}/{doc_info}"
            print(f"Downloading 10-K document: {filing_txt_url}")

            filing_text = requests.get(filing_txt_url, headers=headers).text

            filename = f"10-K_{date}.html"
            with open(filename, 'w', encoding = 'utf-8') as f:
                f.write(filing_text)
            print(f"Saved to {filename}")
            downloaded += 1
        else:
            print("Could not locate 10-K .htm document in index.")

        if downloaded >= num_to_download:
            print(f"\nReached target of {num_to_download} 10-K filings.")
            break
