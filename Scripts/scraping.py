import wikipediaapi
import logging
import json



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


wiki_lt = wikipediaapi.Wikipedia(
    user_agent='LLMDataCollector/1.0 (contact@example.com)',
    language='lt',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


def get_all_writers_recursive(category_name, max_level=2):
    """
    Recursively finds all pages in a category and its subcategories.
    """
    category = wiki_lt.page(f"Category:{category_name}")
    if not category.exists():
        logging.error(f"Category {category_name} not found.")
        return {}

    all_pages = {}

    def walk_category(cat, level):
        if level > max_level:
            return

        logging.info(f"Exploring Category: {cat.title} (Level {level})")

        for member in cat.categorymembers.values():
            # If it's an article, save the text
            if member.ns == wikipediaapi.Namespace.MAIN:
                if member.title not in all_pages:
                    all_pages[member.title] = {
                        "title": member.title,
                        "text": member.text,
                        "url": member.fullurl,
                        "category": cat.title
                    }
                    if len(all_pages) % 50 == 0:
                        logging.info(f"Progress: Collected {len(all_pages)} articles...")


            elif member.ns == wikipediaapi.Namespace.CATEGORY:
                walk_category(member, level + 1)

    walk_category(category, level=0)
    return list(all_pages.values())





target_category = "Lietuvos_rašytojai"


writers_dataset = get_all_writers_recursive(target_category)


file_name = "lithuanian_writers_full.json"
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(writers_dataset, f, ensure_ascii=False, indent=4)

logging.info(f"COMPLETED. Total articles saved: {len(writers_dataset)}")
logging.info(f"File saved as: {file_name}")

