from src.ocrfa import OCR

IMG_TEST = (
    "https://imgs.search.brave.com/vv8k3ywyTr_XUl91lDD8r0tRoW7VOhl6gFJl_m_S57Q/rs:fit:860:0:0/g:ce/aHR0cHM"
    "6Ly91c2Vy/LWltYWdlcy5naXRo/dWJ1c2VyY29udGVu/dC5jb20vMTA3NzQy/MjIvOTI4NzQyMzEt/ODFhNWM0MDAtZjQx/MC0"
    "xMWVhLTkwZjgt/YzAyNjlkOWM3OWFh/LnBuZw"
)


def main():
    ocr = OCR(gpu=False)
    res = ocr.readtext(IMG_TEST)
    print(res)


if __name__ == "__main__":
    main()
