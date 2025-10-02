def build_response(image_b64, angle):
    return {
        "processed_image": image_b64,
        "angle": angle
    }
