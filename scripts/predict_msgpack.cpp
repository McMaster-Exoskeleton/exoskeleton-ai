/**
 * predict_msgpack.cpp
 *
 * Minimal C++ client that sends a single msgpack inference request
 * to the exoskeleton TCN server and prints the predicted torques.
 *
 * Compile:
 *   g++ -O2 -std=c++17 predict_msgpack.cpp -lcurl -o predict_msgpack
 *
 * Run:
 *   ./predict_msgpack
 */

#include <iostream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <msgpack.hpp>

static const char* MSGPACK_URL  = "http://127.0.0.1:8000/predict_msgpack?model=single_joint";
static const int   WINDOW_SIZE  = 187;
static const int   FEATURE_COUNT = 7;  // hip_left only: 6 IMU + 1 joint angle

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

int main() {
    // Build flat input: WINDOW_SIZE * FEATURE_COUNT float32 values (zeroed)
    std::vector<float> flat(WINDOW_SIZE * FEATURE_COUNT, 0.0f);

    // Pack with msgpack
    msgpack::sbuffer buf;
    msgpack::pack(buf, flat);

    // Send POST request
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to init curl\n";
        return 1;
    }

    std::string response_body;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/x-msgpack");

    curl_easy_setopt(curl, CURLOPT_URL, MSGPACK_URL);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buf.data());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)buf.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK || http_code != 200) {
        std::cerr << "Request failed (HTTP " << http_code << "): " << curl_easy_strerror(res) << "\n";
        return 1;
    }

    // Unpack response map
    msgpack::object_handle oh = msgpack::unpack(response_body.data(), response_body.size());
    std::map<std::string, double> result;
    oh.get().convert(result);

    std::cout << "hip_left:     " << result["hip_left"]    << " Nm\n";
    std::cout << "inference_ms: " << result["inference_ms"] << " ms\n";

    return 0;
}
