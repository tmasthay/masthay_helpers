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
    elif len(directive) == 0:
        return None
    else:
        return 'CUSTOM'


def sort_articles(*, articles, directive):
    sort_function = create_sort_function(directive)
    if sort_function is None:
        return articles
    elif sort_function == 'CUSTOM':
        return sorted(
            articles, key=lambda x: directive.index(x.replace('.txt', ''))
        )
    return sorted(articles, key=sort_function)


def main(root):
    with open(f'{root}/LATEX_ORDER.txt', 'r') as file:
        directive = file.read().strip()

    art_dir = f'{root}/articles'
    out_dir = f'{root}/output'
    if not os.path.isdir(art_dir):
        raise ValueError(f'{art_dir} is not a directory')
    os.makedirs(out_dir, exist_ok=True)

    article_files = sort_articles(
        articles=os.listdir(art_dir), directive=directive
    )

    metadata_file_content = ''
    main_file_content = ''
    for article_file in article_files:
        title, author, url, points = process_article_file(
            f'{art_dir}/{article_file}'
        )
        main_file_content += generate_latex_code(title, author, url, points)
        metadata_file_content += generate_metadata_code(title, author, url)

    with open(f'{out_dir}/metadata.tex', 'w') as metadata_file:
        metadata_file.write(metadata_file_content)

    with open(f'{out_dir}/main.tex', 'w') as main_file:
        main_file.write(main_file_content)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='tex')
    args = parser.parse_args()
    args.root = os.abspath(args.root)

    main(args.root)


def process_all(root=None):
    if root is None:
        root = os.getcwd()
    all_dirs = os.listdir(root)
    for directory in all_dirs:
        if os.path.isdir(directory):
            main(directory)


if __name__ == "__main__":
    cli_main()
