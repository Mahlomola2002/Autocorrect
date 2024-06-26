
import pandas as pd
import re
from collections import Counter, defaultdict
from spellchecker import SpellChecker

class Search:
    def __init__(self):
        self.word_frequencies = defaultdict(int)
    
    def switch_letters(self, word):
        return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word) - 1)]
    
    def replace_letter(self, word):
        alphs = 'abcdefghijklmnopqrstuvwxyz0123456789_-#'
        return [word[:i] + l + word[i+1:] for i in range(len(word)) for l in alphs if l != word[i]]
    
    def insert_letter(self, word):
        alphs = 'abcdefghijklmnopqrstuvwxyz0123456789_-#'
        return [word[:i] + l + word[i:] for i in range(len(word) + 1) for l in alphs]
    
    def edits1(self, word, allow_switches=True):
        edits = set()
        if allow_switches:
            edits.update(self.switch_letters(word))
        edits.update(self.replace_letter(word))
        edits.update(self.insert_letter(word))
        return edits
    
    def edits2(self, word, allow_switches=True):
        return {e2 for e1 in self.edits1(word, allow_switches) for e2 in self.edits1(e1, allow_switches)}
    
    def get_corrections(self, word, probs, vocab):
        candidates = {word} | self.edits1(word) | self.edits2(word) & vocab
        sorted_candidates = sorted([(w, probs[w]) for w in candidates if probs.get(w, 0) > 0.00000000000001], key=lambda x: x[1], reverse=True)
        return sorted_candidates or [(word, 0)]
    
    def correct_word_list(self, word_list, probs, vocab):
        return {word: self.get_corrections(word, probs, vocab) for word in word_list}
    
    def preprocess_text(self, text):
        stop_words = {"i", "want", "to", "a", "the", "is", "in", "it", "of", "for", "on", "with", "as", "this", "that"}
        return [word for word in re.sub(r'[^\w\s]', '', text.lower()).split() if word not in stop_words]
    
    def preprocess_text_keep_special(self, text):
        stop_words = {"i", "want", "to", "a", "the", "is", "in", "it", "of", "for", "on", "with", "as", "this", "that"}
        words = text.lower().split(" ")
        return [word for word in words if word.lower() not in stop_words]
    
    def build_inverted_index(self, data):
        inverted_index = defaultdict(list)
        
        for idx, entry in enumerate(data):
            for key in entry:
                if key == "keywords":
                    keywords = entry[key]
                    tokens = [str(k).strip().lower() for k in keywords.split(',') if k.strip()] if isinstance(keywords, str) else []
                elif key != "route":
                    tokens = self.preprocess_text(entry[key])
                else:
                    tokens = []
                
                for token in tokens:
                    if idx not in inverted_index[token]:
                        inverted_index[token].append(idx)
        
        return inverted_index

    
    def phrase_match(self, query, text):
        query_words = query.lower().split()
        text_words = text.lower().split()
        return any(text_words[i:i+len(query_words)] == query_words for i in range(len(text_words) - len(query_words) + 1))
    
    def search_query(self, inverted_index, query, data):
        tokens = self.preprocess_text(query)
        matching_indices = {idx for token in tokens for key in inverted_index.keys() if key.startswith(token) for idx in inverted_index[key]}
        results = []
        
        for idx in matching_indices:
            entry = data[idx]
            score = sum(
                100 if self.phrase_match(query, entry[field]) else (10 if token in entry[field].lower() else (1 if token in entry['keywords'] else 0))
                for token in tokens for field in ['title', 'description', 'blurb']
            )
            results.append((entry, score))
        
        return [entry for entry, _ in sorted(results, key=lambda x: x[1], reverse=True)]
    
    def select_correct_words(self, temp_array, correct):
        selected = []
        
        for word in temp_array:
            if word in correct:
                options = correct[word]
                print(f"Word: {word}")
                
                if any(prob == 0 for _, prob in options):
                    # If probability is 0, suggesting corrections
                    word_1 = word
                    words = SpellChecker().candidates(word_1)
                    
                    if words is not None:
                        words = list(words)  # Convert to list if not None
                        
                        if words:
                            
                            selected.extend(words)  # Append all suggestions to selected
                        else:
                            print(f"No suggestions found for '{word_1}'")
                            selected.append(word_1)
                    else:
                        print(f"No suggestions found for '{word_1}'")
                        selected.append(word_1)
                else:
                    # Sort options based on probability
                    sorted_options = sorted(options, key=lambda x: (self.word_frequencies[x[0]], x[1]), reverse=True)
                    
                    for option, _ in sorted_options:
                        selected.append(option)  # Append all options to selected
        
        return selected
    
    def handle_complex_query(self, query, inverted_index, data):
        subqueries = re.split(r'\band\b|\bor\b', query)
        all_results = []
        
        for subquery in subqueries:
            subquery = subquery.strip()
            result_entries = self.search_query(inverted_index, subquery, data)
            all_results.extend(result_entries)
        
        unique_results = {entry['title']: entry for entry in all_results}.values()
        
        return list(unique_results)
    
    def read_excel_dict(self, file, sheet_name):
        df = pd.read_excel(file, sheet_name=sheet_name)
        data = df.to_dict(orient='records')
        processed_data = [
        {
            "title": str(row.get("Title", "")).lower(),
            "route": str(row.get("Route", "")).lower() if pd.notna(row.get("Route")) else '',
            "description": str(row.get("Description", "")).lower(),
            "blurb": str(row.get("Blurb", "")).lower(),
            "keywords": [k.strip().lower() for k in str(row.get("Keywords", "")).split(',')]
        }
        for row in data
        ]
        return processed_data
        
        
    
    def suggest_words(self, query, pro, vocab):
        words_to_correct = self.preprocess_text(query)
        correct_list=self.correct_word_list(words_to_correct, pro, vocab)
        correct = self.select_correct_words(words_to_correct,correct_list)
        return {query:correct}


    



# Example usage


spell = Search()

data_dict = spell.read_excel_dict('data2.xlsx', 'Version 2')


combined_text_list = [
    ' '.join([
        entry.get('title', ''),
        entry.get('description', ''),
        entry.get('blurb', ''),
        ' '.join([re.sub(r'[^\w\s]', '', k.lower()) for k in entry.get("keywords", [])])
        
    ])
    for entry in data_dict
]
combined_text = ' '.join(combined_text_list)
words = re.findall(r'\w+', combined_text)
vocab = set(words)
word_count = Counter(words)
total_words = sum(word_count.values())

probs = {word: count / total_words for word, count in word_count.items()}


inverted_index = spell.build_inverted_index(data_dict)
while True:
    text=str(input("Enter the query:\n"))
    print(spell.suggest_words(text,probs,vocab))
"""
while True:
    
    query = input("Enter a query: ").lower()
    if query == "quit":
        print("Quit application")
        break
    
    result_entries = spell.handle_complex_query(query, inverted_index, data_dict)
    print(f"Entries matching before correction '{query}':")
    for entry in result_entries:
        print(entry)
    
    if len(result_entries)!=0:
        selection = input("Enter the number of the entry you want to see or 'quit': ")
        try:
            selection_int = int(selection)
            if 1 <= selection_int <= len(result_entries):
                selected_entry = result_entries[selection_int - 1]
                print(f"\nSelected Entry:\n{selected_entry}")
            else:
                print("Invalid selection number.")
        except ValueError:
            if selection.lower() == "quit":
                break
            else:
                print("Invalid input. Please enter a number or 'quit'.")
    
    inp = input("Do you want to correct your input? (Y/N): ").lower()
    if inp == "y":
        temp_array = spell.preprocess_text(query)
        correct = spell.correct_word_list(temp_array, probs, vocab)
        
        selected = spell.select_correct_words(temp_array, correct)
        
        corrected_query = " ".join(selected)
        result_entries = spell.handle_complex_query(corrected_query, inverted_index, data_dict)
        
        print(f"Entries matching after correction '{corrected_query}':")
        for entry in result_entries:
            print(entry)
"""