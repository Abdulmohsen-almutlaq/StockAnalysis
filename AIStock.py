import asyncio
import base64
import json
import os
import requests
from pyppeteer import launch
from PIL import Image
from tradingview_ta import TA_Handler, Interval

# ===========================
# Configuration Parameters
# ===========================

# OpenAI API Key
api_key ="your api"
# Paths to Chrome executable (modify if different)
chrome_executable_path = 'C:\Program Files\Google\Chrome\Application\chrome.exe'

# Directory paths
screenshots_dir = "screenshots"
indicators_dir = "indicators"

# Ensure directories exist
os.makedirs(screenshots_dir, exist_ok=True)
os.makedirs(indicators_dir, exist_ok=True)

# Load exchange information from a JSON file
with open('exchange_info.json', 'r') as f:
    exchange_info = json.load(f)

# ===========================
# User Input for Symbols and Timeframes
# ===========================

# Allow user to choose the stock symbol
stock_symbols = input("Enter the stock symbols you want to capture (comma separated, e.g., AAPL,TSLA): ").split(",")

# Get exchange for each symbol from the JSON file
stock_symbols_encoded = []
for symbol in stock_symbols:
    symbol = symbol.strip().upper()
    exchange = exchange_info.get(symbol, "NASDAQ")  # Default to NASDAQ if not found
    stock_symbols_encoded.append(f"{exchange}%3A{symbol}")

# Allow user to choose the timeframes
timeframe_options = [
    {"label": "1 minute", "save_name": "1m"},
    {"label": "1 Day", "save_name": "1D"},
    {"label": "5 minutes", "save_name": "5m"},
    {"label": "15 minutes", "save_name": "15m"},
    {"label": "30 minutes", "save_name": "30m"},
    {"label": "1 hour", "save_name": "1h"},
    {"label": "4 hours", "save_name": "4h"},
    {"label": "1 week", "save_name": "1W"}
]

# Display available timeframes
print("Available timeframes:")
for i, tf in enumerate(timeframe_options):
    print(f"{i + 1}. {tf['label']}")

# Allow user to select multiple timeframes by index
selected_indices = input("Enter the indices of the timeframes you want to use (comma separated, e.g., 1,2,3): ").split(",")
timeframes = [timeframe_options[int(index.strip()) - 1] for index in selected_indices]

# ===========================
# Helper Functions
# ===========================

def encode_image(image_path):
    """
    Encodes an image to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_indicator_readings(symbol, exchange, interval):
    """
    Fetches technical indicator readings for a given stock symbol.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        exchange (str): Exchange name (e.g., "NASDAQ").
        interval (str): Timeframe interval (e.g., Interval.INTERVAL_1_MINUTE).

    Returns:
        dict: Dictionary containing indicator readings and analysis summary.
    """
    try:
        handler = TA_Handler(
            symbol=symbol,
            screener="america",
            exchange=exchange,
            interval=interval
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators

        return {
            "indicators": indicators,
        }
    except Exception as e:
        print(f"Error fetching indicators for {symbol}: {e}")
        return {}

def crop_screenshot(image_path):
    """
    Crops the screenshot to a specific size.

    Args:
        image_path (str): Path to the image file.
    """
    try:
        image = Image.open(image_path)

        final_width = 1560
        final_height = 1080

        crop_area = (0, 0, final_width, final_height)

        cropped_image = image.crop(crop_area)

        cropped_image.save(image_path)
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")

# ===========================
# OpenAI API Interaction
# ===========================
def get_user_prompt_choice():
    print("Choose the type of analysis you want:")
    print("1. Simple with price (limit price and stop loss)")
    print("2. In-depth analysis")
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")

def send_to_openai(symbol, timeframe_label, image_path, indicator_readings):
    """
    Sends the image and indicator readings to the OpenAI API for analysis.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        timeframe_label (str): Timeframe label (e.g., "1 minute").
        image_path (str): Path to the screenshot image.
        indicator_readings (dict): Technical indicator readings.

    Returns:
        dict: Response from the OpenAI API.
    """
    # Encode the image
    base64_image = encode_image(image_path)

    # Format the indicator readings as a readable string
    readings_text = "\n".join([f"{key}: {value}" for key, value in indicator_readings.get("indicators", {}).items()])

    # Get user's choice for the prompt
    prompt_choice = get_user_prompt_choice()

    if prompt_choice == '1':
        prompt = f"give me  -limit price -stop lost\nyour response shouldn't be very long\nAct as an experienced day trader. Your objective is to analyze the price and volume{timeframe_label} {readings_text}\n also candlestick patterns based on this also the image down there   identify potential buying or selling opportunities. Utilize advanced charting tools and technical indicators to scrutinize both short-term and long-term patterns, taking into account historical data and recent market movements. Assess the correlation between price and volume to gauge the strength or weakness of a particular price trend. Provide a comprehensive analysis report that details potential breakout or breakdown points, support and resistance levels, and any anomalies or divergences noticed. Your analysis should be backed by logical reasoning and should include potential risk and reward scenarios. Always adhere to best practices in technical analysis and maintain the highest standards of accuracy and objectivity.< this is the thinking part. \nyou give me PRICES and short summary. \nyour response shouldn't be very long"
    else:
        prompt = f"your response shouldn't be very long\nAct as an experienced day trader. Your objective is to analyze the price and volume{timeframe_label} {readings_text}\n also candlestick patterns based on this also the image down there   identify potential buying or selling opportunities. Utilize advanced charting tools and technical indicators to scrutinize both short-term and long-term patterns, taking into account historical data and recent market movements. Assess the correlation between price and volume to gauge the strength or weakness of a particular price trend. Provide a comprehensive analysis report that details potential breakout or breakdown points, support and resistance levels, and any anomalies or divergences noticed. Your analysis should be backed by logical reasoning and should include potential risk and reward scenarios. Always adhere to best practices in technical analysis and maintain the highest standards of accuracy and objectivity.< this is the thinking part. \nyou give me PRICES and short summary. \nyour response shouldn't be very long"

    # Prepare the API request payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",  # Ensure this model supports image inputs
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 600
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for {symbol} ({timeframe_label}): {http_err}")
    except Exception as err:
        print(f"Other error occurred for {symbol} ({timeframe_label}): {err}")
    return {}
async def capture_screenshot_and_fetch_indicators(symbol_encoded, timeframe_label, save_as, indicators_save_path, openai_response_path, browser):
    """
    Captures a screenshot of the TradingView chart, fetches technical indicators,
    and sends both to the OpenAI API.

    Args:
        symbol_encoded (str): Encoded symbol string (e.g., "NASDAQ%3AAAPL").
        timeframe_label (str): Timeframe label (e.g., "1 minute").
        save_as (str): Filename to save the screenshot.
        indicators_save_path (str): Path to save the indicator readings.
        openai_response_path (str): Path to save the OpenAI API response.
        browser: Pyppeteer browser instance.
    """
    # Decode the symbol
    try:
        exchange, symbol = symbol_encoded.split("%3A")
    except ValueError:
        print(f"Invalid symbol format: {symbol_encoded}")
        return

    # Map timeframe_label to tradingview-ta interval
    interval_mapping = {
        "1 minute": Interval.INTERVAL_1_MINUTE,
        "1 day": Interval.INTERVAL_1_DAY,
        "5 minutes": Interval.INTERVAL_5_MINUTES,
        "15 minutes": Interval.INTERVAL_15_MINUTES,
        "30 minutes": Interval.INTERVAL_30_MINUTES,
        "1 hour": Interval.INTERVAL_1_HOUR,
        "4 hours": Interval.INTERVAL_4_HOURS,
        "1 week": Interval.INTERVAL_1_WEEK,
    }

    ta_interval = interval_mapping.get(timeframe_label.lower())
    if not ta_interval:
        print(f"Unsupported timeframe: {timeframe_label} for symbol: {symbol}")
        return

    # Open TradingView chart and capture screenshot
    try:
        page = await browser.newPage()
        await page.setViewport({"width": 1920, "height": 1080})

        chart_url = f'https://www.tradingview.com/chart/?symbol={symbol_encoded}'
        await page.goto(chart_url, {'waitUntil': 'networkidle2'})
        await asyncio.sleep(2)  # Wait for the page to load

        # Capture screenshot
        try:
            await page.screenshot({'path': save_as, 'fullPage': True})
            await page.close()
        except Exception as e:
            print(f"Error capturing screenshot for {symbol} ({timeframe_label}): {e}")
            await page.close()
            return

        # Crop the screenshot
        crop_screenshot(save_as)

        # Fetch indicator readings
        indicator_readings = get_indicator_readings(symbol, exchange, ta_interval)

        # Save indicator readings to JSON
        try:
            with open(indicators_save_path, 'w') as f:
                json.dump(indicator_readings, f, indent=4)
        except Exception as e:
            print(f"Error saving indicators for {symbol} ({timeframe_label}): {e}")

        # Send data to OpenAI
        openai_response = send_to_openai(symbol, timeframe_label, save_as, indicator_readings)
        # Save OpenAI response to JSON
        try:
            with open(openai_response_path, 'w') as f:
                json.dump(openai_response, f, indent=4)
        except Exception as e:
            print(f"Error saving OpenAI response for {symbol} ({timeframe_label}): {e}")

        print(f"Processed {symbol} ({timeframe_label}) successfully.")

    except Exception as e:
        print(f"Unexpected error for {symbol} ({timeframe_label}): {e}")

# ===========================
# Main Function
# ===========================

async def main():
    # Launch a single browser instance for all tasks
    browser = await launch(headless=True, executablePath=chrome_executable_path)

    tasks = []
    openai_response_paths = []

    for symbol_encoded in stock_symbols_encoded:
        for tf in timeframes:
            symbol_clean = symbol_encoded.split("%3A")[1]
            timeframe_label = tf["label"]
            save_as = os.path.join(screenshots_dir, f'tradingview_chart_{symbol_clean}_{tf["save_name"]}.png')
            indicators_save_path = os.path.join(indicators_dir, f'{symbol_clean}_{tf["save_name"]}_indicators.json')
            openai_response_path = os.path.join(indicators_dir,
                                                f'{symbol_clean}_{tf["save_name"]}_openai_response.json')

            openai_response_paths.append(openai_response_path)

            tasks.append(
                capture_screenshot_and_fetch_indicators(
                    symbol_encoded=symbol_encoded,
                    timeframe_label=timeframe_label,
                    save_as=save_as,
                    indicators_save_path=indicators_save_path,
                    openai_response_path=openai_response_path,
                    browser=browser
                )
            )

    # Run all tasks concurrently with limited concurrency to manage resources
    semaphore = asyncio.Semaphore(5)

    async def sem_task(task):
        async with semaphore:
            await task

    await asyncio.gather(*(sem_task(task) for task in tasks))

    # Close the browser after all tasks are done
    await browser.close()

    # Process and print OpenAI responses with improved error handling
    for response_path in openai_response_paths:
        try:
            with open(response_path, 'r') as f:
                openai_response = json.load(f)


                # Check if 'choices' key exists and has content
                if 'choices' in openai_response and openai_response['choices']:
                    print(openai_response["choices"][0]["message"]["content"])
                else:
                    # If 'choices' is missing or empty, print the entire response for debugging
                    print("Unexpected response format. Full response:")
                    print(json.dumps(openai_response, indent=2))

                    # Check for error messages
                    if 'error' in openai_response:
                        print(f"Error message: {openai_response['error'].get('message', 'Unknown error')}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file {response_path}. The file may be empty or contain invalid JSON.")
        except Exception as e:
            print(f"Error reading file {response_path}: {e}")

    print("\n\n\n\nAll screenshots captured, indicator readings fetched, and OpenAI responses processed.")
if __name__ == "__main__":
    asyncio.run(main())