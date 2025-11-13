import os

def file_read(main_folder : str = "docs") -> list :
    """
    Returns a list containing data from files to fulfil the requirements of current API
    txt files should have 'department','year' and a blank line before main content
    """

    content = []
    for dirpath, _, filenames in os.walk(main_folder):
        for name in filenames:
            if name == 'README.txt' :
                continue
            if name[-3:].lower() != "txt" :
                continue
            filepath = os.path.join(dirpath,name)
            with open(filepath,'r') as doc_file:
                dep_name = doc_file.readline().split(":")[1].strip()
                year = doc_file.readline().split(":")[1].strip()
                doc_file.readline()
                file_body = doc_file.read()
                entry = {'content' : file_body , 'metadata': {'department':dep_name, 'year':year}}
            content.append(entry)
    return content


if __name__ == "__main__":
    file_read()