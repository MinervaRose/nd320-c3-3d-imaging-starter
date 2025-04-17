import os
import sys
import datetime
import time
import shutil
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pydicom
from PIL import Image, ImageFont, ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]
    hdr = dcmlist[0]
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)


def get_predicted_volumes(pred):
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)
    total_volume = np.sum(pred > 0)
    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}


def create_report(inference, header, orig_vol, pred_vol):
    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)

    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text(
        (10, 90),
        f"Patient ID: {header.PatientID}\n"
        f"Study Description: {header.StudyDescription}\n"
        f"Study Date: {header.StudyDate}\n"
        f"Series Description: {header.SeriesDescription}\n"
        f"Modality: {header.Modality}\n"
        f"Image Type: {header.ImageType}\n"
        f"Anterior Volume: {inference['anterior']}\n"
        f"Posterior Volume: {inference['posterior']}\n"
        f"Total Volume: {inference['total']}\n",
        (255, 255, 255),
        font=main_font
    )

    # Original image slice
    nd_orig = np.flip((orig_vol[0, :, :] / np.max(orig_vol[0, :, :])) * 255).T.astype(np.uint8)
    pil_orig = Image.fromarray(nd_orig, mode="L").convert("RGBA").resize((400, 400))
    pimg.paste(pil_orig, box=(50, 500))

    # Prediction mask slice
    nd_pred = np.flip((pred_vol[0, :, :] / np.max(pred_vol[0, :, :])) * 255).T.astype(np.uint8)
    pil_pred = Image.fromarray(nd_pred, mode="L").convert("RGBA").resize((400, 400))
    pimg.paste(pil_pred, box=(550, 500))

    return pimg


def save_report_as_dcm(header, report, path):
    out = pydicom.Dataset(header)
    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    out.is_little_endian = True
    out.is_implicit_VR = False

    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID

    out.Modality = "OT"
    out.SeriesDescription = "HippoVolume.AI"
    out.Rows = report.height
    out.Columns = report.width
    out.ImageType = r"DERIVED\PRIMARY\AXIAL"
    out.SamplesPerPixel = 3
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0
    out.BitsAllocated = 8
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm
    out.ImagesInAcquisition = 1
    out.WindowCenter = ""
    out.WindowWidth = ""
    out.BurnedInAnnotation = "YES"
    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)


def get_series_for_inference(path):
    dicoms = []
    print("\n--- Scanning DICOM files in:", path)

    for f in os.listdir(path):
        if f.endswith(".dcm"):
            dcm = pydicom.dcmread(os.path.join(path, f))
            print("Found SeriesDescription:", dcm.SeriesDescription)
            dicoms.append(dcm)

    # Filter by HippoCrop
    series_for_inference = [d for d in dicoms if d.SeriesDescription == "HippoCrop"]

    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: Cannot determine a single series to run inference on.")
        return []

    return series_for_inference

def os_command(command):
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # If path contains DICOM files directly, use it
    if any(f.endswith(".dcm") for f in os.listdir(sys.argv[1])):
        study_dir = sys.argv[1]
    else:
        subdirs = [
            os.path.join(sys.argv[1], d)
            for d in os.listdir(sys.argv[1])
            if os.path.isdir(os.path.join(sys.argv[1], d))
        ]
        if not subdirs:
            print("No subdirectories found. Exiting.")
            sys.exit(1)
        study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path="/home/workspace/model.pth"
    )
    print("Input volume shape:", volume.shape)
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    print("Unique values in prediction:", np.unique(pred_label))
    pred_volumes = get_predicted_volumes(pred_label)

    print("Creating and pushing report...")
    report_save_path = "../out/report.dcm"
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    os_command("storescu localhost 4242 -v -aec HIPPOAI +r +sd /home/workspace/out/report.dcm")

    time.sleep(2)
    #shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(
        f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape} ",
        f"volume ant: {pred_volumes['anterior']}, ",
        f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}"
    )
