import os
import argparse


def process_article_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    title = lines[0].strip()
    author = lines[1].strip()
    url = lines[2].strip()
    points = lines[3:]
    return title, author, url, points


def generate_latex_code(title, author, url, points):
    latex_code = (
        r"\section{%s by %s}\label{%s}\href{%s}{\nameref*{%s}}"
        % (title, author, title.replace(' ', ''), url, title.replace(' ', ''))
        + '\n'
    )
    for point in points:
        if point.startswith('*'):
            latex_code += r"    \item %s" % point[1:].strip() + '\n'
        else:
            latex_code += r"\item %s" % point.strip() + '\n'
    return latex_code


def generate_metadata_code(title, author, url):
    command_name = title.replace(' ', '')
    metadata_code = (
        r"\newcommand{\data%s}[1]{\ifcase#1 %s\or %s\or %s\fi}"
        % (command_name, title, author, url)
        + '\n'
    )
    return metadata_code


def create_sort_function(directive):
    if directive == 'ALPHANUMERIC':
        return lambda x: x
    elif directive == 'ALPHABETICAL':
        return lambda x: x
    elif directive == 'CHRONOLOGICAL':
        return lambda x: x
    elif directive == 'REVERSE_CHRONOLOGICAL':
        return lambda x: x
    elif directive == 'RANDOM':
        return lambda x: x
    else:
        return None


def sort_articles(*, articles, directive):
    sort_function = create_sort_function(directive)
    if sort_function is None:
        return articles
    return sorted(articles, key=sort_function)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='tex')
    args = parser.parse_args()
    args.root = os.abspath(args.root)

    with open(f'{args.root}/LATEX_ORDER.txt', 'r') as file:
        directive = file.read().strip()

    article_files = sort_articles(
        articles=os.listdir('articles'), directive=directive
    )

    metadata_file_content = ''
    main_file_content = ''
    for article_file in article_files:
        title, author, url, points = process_article_file(
            f'articles/{article_file}'
        )
        main_file_content += generate_latex_code(title, author, url, points)
        metadata_file_content += generate_metadata_code(title, author, url)

    os.makedirs(f'{args.root}/output', exist_ok=True)
    with open(f'{args.root}/output/metadata.tex', 'w') as metadata_file:
        metadata_file.write(metadata_file_content)

    with open(f'{args.root}/output/main.tex', 'w') as main_file:
        main_file.write(main_file_content)


if __name__ == "__main__":
    main()
