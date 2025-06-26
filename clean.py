import re

def clean_text_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='surrogateescape') as infile:
        content = infile.read()

    # إزالة surrogate characters غير الصالحة
    cleaned_content = re.sub(r'[\ud800-\udfff]', '', content)

    with open(output_path, 'w', encoding='utf-8', errors='ignore') as outfile:
        outfile.write(cleaned_content)

# الاستخدام:
clean_text_file("vault.txt", "vault_cleaned.txt")
