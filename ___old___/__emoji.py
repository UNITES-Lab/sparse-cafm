#!/usr/bin/env python3
import sys

def byte_to_variation_selector(b):
    """
    Converts a single byte (an integer 0–255) into a Unicode variation selector.
    For bytes 0-15, we use U+FE00 to U+FE0F.
    For bytes 16-255, we use U+E0100 to U+E01EF.
    """
    if b < 16:
        return chr(0xFE00 + b)
    else:
        return chr(0xE0100 + (b - 16))

def encode_message(message, base='😊'):
    """
    Encodes a message string into an emoji that carries hidden data.
    The base emoji is printed normally and the message is encoded into
    a sequence of invisible variation selectors appended to it.
    """
    # Convert the string into bytes using UTF-8 encoding.
    message_bytes = message.encode('utf-8')
    
    # Start with the base emoji.
    encoded = [base]
    
    # Append a variation selector for each byte.
    for b in message_bytes:
        encoded.append(byte_to_variation_selector(b))
        
    return ''.join(encoded)

def decode_message(emoji_with_data):
    """
    (Optional) Decodes the hidden message from an emoji produced by encode_message.
    It assumes that the first character is the base and subsequent variation selectors
    represent the encoded bytes.
    """
    def variation_selector_to_byte(ch):
        code = ord(ch)
        if 0xFE00 <= code <= 0xFE0F:
            return code - 0xFE00
        elif 0xE0100 <= code <= 0xE01EF:
            return code - 0xE0100 + 16
        else:
            return None

    # Skip the base character.
    data_bytes = []
    for ch in emoji_with_data[1:]:
        byte_val = variation_selector_to_byte(ch)
        if byte_val is None:
            # Stop if we hit a character that isn't a variation selector.
            break
        data_bytes.append(byte_val)
    try:
        return bytes(data_bytes).decode('utf-8')
    except UnicodeDecodeError:
        return None

def main():
    if len(sys.argv) < 2:
        # If no command-line arguments are given, prompt the user.
        message = input("Enter the message to encode: ")
    else:
        # Join all arguments into one message (in case there are spaces).
        message = " ".join(sys.argv[1:])
    
    encoded_emoji = encode_message(message)
    print("Encoded emoji:")
    print(encoded_emoji)
    
    # (Optional) Show that we can recover the message.
    decoded_message = decode_message(encoded_emoji)
    if decoded_message is not None:
        print("\nDecoded message (for verification):")
        print(decoded_message)
    else:
        print("\nWarning: Could not decode the message.")

if __name__ == '__main__':
    main()
