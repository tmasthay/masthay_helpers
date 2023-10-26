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


def generate_latex_code(title, author, url, points, list_type):
    latex_code = (
        r"\section{%s by %s}\label{%s}\href{%s}{\nameref*{%s}}"
        % (title, author, title.replace(' ', ''), url, title.replace(' ', ''))
        + '\n'
    )
    latex_code += r"\begin{%s}" % list_type + '\n'
    for point in points:
        point = point.strip()
        if point.strip().startswith('*'):
            point = point.replace('*', '').strip()
            latex_code += r"    \begin{%s}" % list_type + '\n'
            latex_code += r"    \item %s" % point + '\n'
            latex_code += r"    \end{%s}" % list_type + '\n'
        else:
            latex_code += r"\item %s" % point.strip() + '\n'
    latex_code += r"\end{%s}" % list_type + '\n\n'
    return latex_code


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


def main(root, list_type='enumerate', verbose=False):
    def main_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    main_print(f'Processing {root}...', end='')
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

    main_file_content = ''
    for article_file in article_files:
        title, author, url, points = process_article_file(
            f'{art_dir}/{article_file}'
        )
        main_file_content += generate_latex_code(
            title, author, url, points, list_type
        )

    with open(f'{out_dir}/main.tex', 'w') as main_file:
        main_file.write(main_file_content)
    main_print('SUCCESS')


def process_all(root=None, list_type='enumerate', verbose=False):
    if root is None:
        root = os.getcwd()
    all_dirs = os.listdir(root)
    for directory in all_dirs:
        if os.path.isdir(directory):
            main(directory, list_type, verbose)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=os.getcwd())
    parser.add_argument('-l', '--list_type', type=str, default='enumerate')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)

    process_all(args.root, args.list_type, args.verbose)


if __name__ == "__main__":
    cli_main()
