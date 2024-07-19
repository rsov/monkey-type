use std::collections::VecDeque;
use std::error::Error;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
#[allow(unused)]
use rten_tensor::prelude::*;

struct Args {
    image: String,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Usage: {bin_name} <image>",
                    bin_name = parser.bin_name().unwrap_or("hello_ocrs")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let image = values.pop_front().ok_or("missing `image` arg")?;

    Ok(Args { image })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let now = chrono::offset::Local::now();

    // Use the `download-models.sh` script to download the models.
    let detection_model_path = "./text-detection.rten";
    let rec_model_path = "./text-recognition.rten";

    println!("Loading models..");
    let detection_model =Model::load_file(detection_model_path);
    let recognition_model = Model::load_file(rec_model_path);

    
    if detection_model.is_err() {
        panic!("Cannot open {:?}",  detection_model_path)
    }

    if recognition_model.is_err() {
        panic!("Cannot open {:?}",  rec_model_path)
    }

    println!("Starting engine..");
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model.unwrap()),
        recognition_model: Some(recognition_model.unwrap()),
        ..Default::default()
    })?;

    // Read image using image-rs library, and convert to RGB if not already
    // in that format.
    let img = image::open(&args.image).map(|image| image.into_rgb8());

    if img.is_err() {
        panic!("Image not found at {:?}", args.image);
    }

    let img =img.unwrap();

    

    // Apply standard image pre-processing expected by this library (convert
    // to greyscale, map range to [-0.5, 0.5]).
    let img_source = ImageSource::from_bytes(img.as_raw(), img.dimensions())?;
    let ocr_input = engine.prepare_input(img_source)?;

    // Detect and recognize text. If you only need the text and don't need any
    // layout information, you can also use `engine.get_text(&ocr_input)`,
    // which returns all the text in an image as a single string.

    println!("Detecting words..");

    // Get oriented bounding boxes of text words in input image.
    let word_rects = engine.detect_words(&ocr_input)?;


    println!("Finding text lines..");
    // Group words into lines. Each line is represented by a list of word
    // bounding boxes.
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);

    // Recognize the characters in each line.
    println!("Recognizing text..");
    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

    println!("\n");
    for line in line_texts
        .iter()
        .flatten()
        // Filter likely spurious detections. With future model improvements
        // this should become unnecessary.
        .filter(|l| l.to_string().len() > 1)
    {
        println!("{}", line);
    }

    let diff = chrono::offset::Local::now() - now;
    println!("\n");
    println!("Took time: {:?}s", diff.num_seconds());
    Ok(())
}