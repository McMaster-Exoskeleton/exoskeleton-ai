/**
 * test_endpoint.cpp
 *
 * C++ test client for the exoskeleton TCN inference server.
 * Tests both JSON and msgpack endpoints and compares latency.
 *
 * Dependencies (install on Pi):
 *   sudo apt install libcurl4-openssl-dev nlohmann-json3-dev libmsgpack-dev
 *
 * Compile:
 *   g++ -O2 -std=c++17 test_endpoint.cpp -lcurl -o test_endpoint
 *
 * Run:
 *   ./test_endpoint
 *   ./test_endpoint --runs 50
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <msgpack.hpp>
#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Config — must match server.py
// ---------------------------------------------------------------------------

static const char*  HEALTH_URL      = "http://127.0.0.1:8000/health";
static const char*  JSON_URL        = "http://127.0.0.1:8000/predict";
static const char*  MSGPACK_URL     = "http://127.0.0.1:8000/predict_msgpack";
static const int    WINDOW_SIZE     = 187;
static const int    FEATURE_COUNT   = 28;
static const int    DEFAULT_RUNS    = 20;
static const double BUDGET_MS       = 10.0;  // 100 Hz

// ---------------------------------------------------------------------------
// libcurl helpers
// ---------------------------------------------------------------------------

struct CurlResponse {
    std::string body;
    long        http_code = 0;
};

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* resp = static_cast<CurlResponse*>(userdata);
    resp->body.append(ptr, size * nmemb);
    return size * nmemb;
}

static CurlResponse http_get(CURL* curl, const std::string& url) {
    CurlResponse resp;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.http_code);
    return resp;
}

static CurlResponse http_post(CURL* curl, const std::string& url,
                               const char* data, size_t size, const char* content_type) {
    CurlResponse resp;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, content_type);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)size);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.http_code);

    curl_slist_free_all(headers);
    return resp;
}

// ---------------------------------------------------------------------------
// JSON request/response
// ---------------------------------------------------------------------------

static std::string build_json_request(const std::vector<float>& flat) {
    nlohmann::json window = nlohmann::json::array();
    for (int t = 0; t < WINDOW_SIZE; ++t) {
        nlohmann::json row = nlohmann::json::array();
        for (int f = 0; f < FEATURE_COUNT; ++f) {
            row.push_back(flat[t * FEATURE_COUNT + f]);
        }
        window.push_back(row);
    }
    return nlohmann::json{{"window", window}}.dump();
}

struct Prediction {
    double hip_left, hip_right, knee_left, knee_right, inference_ms;
};

static Prediction parse_json_response(const std::string& body) {
    auto j = nlohmann::json::parse(body);
    return {
        j["hip_left"].get<double>(),
        j["hip_right"].get<double>(),
        j["knee_left"].get<double>(),
        j["knee_right"].get<double>(),
        j["inference_ms"].get<double>(),
    };
}

// ---------------------------------------------------------------------------
// Msgpack request/response
// The server expects: flat array of WINDOW_SIZE * FEATURE_COUNT float32 values
// The server returns: map with hip_left, hip_right, knee_left, knee_right, inference_ms
// ---------------------------------------------------------------------------

static std::string build_msgpack_request(const std::vector<float>& flat) {
    msgpack::sbuffer buf;
    msgpack::pack(buf, flat);
    return std::string(buf.data(), buf.size());
}

static Prediction parse_msgpack_response(const std::string& body) {
    msgpack::object_handle oh = msgpack::unpack(body.data(), body.size());
    msgpack::object obj = oh.get();

    // Response is a map — convert to std::map to look up keys
    std::map<std::string, double> result;
    obj.convert(result);

    return {
        result["hip_left"],
        result["hip_right"],
        result["knee_left"],
        result["knee_right"],
        result["inference_ms"],
    };
}

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size())) - 1;
    return v[std::min(idx, v.size() - 1)];
}

static double vmin(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
static double vmax(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

static std::string fmt(double v) {
    std::ostringstream s;
    s.precision(1);
    s << std::fixed << v;
    return s.str();
}

static void print_table_row(const char* label,
                             double json_rt, double mp_rt,
                             double inference,
                             double json_oh, double mp_oh) {
    printf("  %-8s %10s  %10s  %10s  %10s  %10s  ms\n",
           label,
           fmt(json_rt).c_str(), fmt(mp_rt).c_str(),
           fmt(inference).c_str(),
           fmt(json_oh).c_str(), fmt(mp_oh).c_str());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int num_runs = DEFAULT_RUNS;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--runs" && i + 1 < argc) {
            num_runs = std::stoi(argv[++i]);
        }
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to init curl\n";
        return 1;
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "Exoskeleton C++ Endpoint Test\n";
    std::cout << std::string(60, '=') << "\n";

    // ------------------------------------------------------------------
    // [1] Health check
    // ------------------------------------------------------------------
    std::cout << "\n[1] Health check...\n";
    auto health_resp = http_get(curl, HEALTH_URL);
    if (health_resp.http_code != 200) {
        std::cerr << "  Server not ready (HTTP " << health_resp.http_code << ")\n";
        std::cerr << "  Is the server running?\n";
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        return 1;
    }
    std::cout << "  OK\n";

    // ------------------------------------------------------------------
    // [2] Single prediction — JSON, zero input
    // ------------------------------------------------------------------
    std::cout << "\n[2] Single prediction — JSON (zero input)...\n";
    {
        std::vector<float> flat(WINDOW_SIZE * FEATURE_COUNT, 0.0f);
        std::string body = build_json_request(flat);
        auto resp = http_post(curl, JSON_URL, body.c_str(), body.size(),
                              "Content-Type: application/json");
        if (resp.http_code != 200) {
            std::cerr << "  Failed (HTTP " << resp.http_code << "): " << resp.body << "\n";
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            return 1;
        }
        auto p = parse_json_response(resp.body);
        printf("  hip_left:   %f Nm/kg\n", p.hip_left);
        printf("  hip_right:  %f Nm/kg\n", p.hip_right);
        printf("  knee_left:  %f Nm/kg\n", p.knee_left);
        printf("  knee_right: %f Nm/kg\n", p.knee_right);
        printf("  inference:  %.3f ms\n",  p.inference_ms);
    }

    // ------------------------------------------------------------------
    // [3] Single prediction — msgpack, zero input
    // ------------------------------------------------------------------
    std::cout << "\n[3] Single prediction — msgpack (zero input)...\n";
    {
        std::vector<float> flat(WINDOW_SIZE * FEATURE_COUNT, 0.0f);
        std::string body = build_msgpack_request(flat);
        auto resp = http_post(curl, MSGPACK_URL, body.c_str(), body.size(),
                              "Content-Type: application/x-msgpack");
        if (resp.http_code != 200) {
            std::cerr << "  Failed (HTTP " << resp.http_code << "): " << resp.body << "\n";
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            return 1;
        }
        auto p = parse_msgpack_response(resp.body);
        printf("  hip_left:   %f Nm/kg\n", p.hip_left);
        printf("  hip_right:  %f Nm/kg\n", p.hip_right);
        printf("  knee_left:  %f Nm/kg\n", p.knee_left);
        printf("  knee_right: %f Nm/kg\n", p.knee_right);
        printf("  inference:  %.3f ms\n",  p.inference_ms);
    }

    // ------------------------------------------------------------------
    // [4] Latency benchmark — JSON vs msgpack
    // ------------------------------------------------------------------
    std::cout << "\n[4] Latency benchmark — JSON vs msgpack (" << num_runs << " runs each)...\n";
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> flat(WINDOW_SIZE * FEATURE_COUNT);
        for (auto& v : flat) v = dist(rng);

        std::string json_body    = build_json_request(flat);
        std::string msgpack_body = build_msgpack_request(flat);

        std::vector<double> json_rt, json_inf;
        std::vector<double> mp_rt,   mp_inf;

        std::cout << "  JSON...\n";
        for (int i = 0; i < num_runs; ++i) {
            auto t0   = std::chrono::high_resolution_clock::now();
            auto resp = http_post(curl, JSON_URL, json_body.c_str(), json_body.size(),
                                  "Content-Type: application/json");
            auto t1   = std::chrono::high_resolution_clock::now();
            json_rt.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            json_inf.push_back(parse_json_response(resp.body).inference_ms);
            if ((i + 1) % 5 == 0) std::cout << "  " << (i+1) << "/" << num_runs << "\n";
        }

        std::cout << "  msgpack...\n";
        for (int i = 0; i < num_runs; ++i) {
            auto t0   = std::chrono::high_resolution_clock::now();
            auto resp = http_post(curl, MSGPACK_URL, msgpack_body.c_str(), msgpack_body.size(),
                                  "Content-Type: application/x-msgpack");
            auto t1   = std::chrono::high_resolution_clock::now();
            mp_rt.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            mp_inf.push_back(parse_msgpack_response(resp.body).inference_ms);
            if ((i + 1) % 5 == 0) std::cout << "  " << (i+1) << "/" << num_runs << "\n";
        }

        // Overhead = round-trip minus inference
        std::vector<double> json_oh, mp_oh;
        for (int i = 0; i < num_runs; ++i) {
            json_oh.push_back(json_rt[i] - json_inf[i]);
            mp_oh.push_back(mp_rt[i] - mp_inf[i]);
        }

        std::cout << "\n";
        printf("  %-8s %10s  %10s  %10s  %10s  %10s\n",
               "", "JSON rt", "MP rt", "inference", "JSON oh", "MP oh");
        printf("  %-8s %10s  %10s  %10s  %10s  %10s\n",
               "", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)");
        std::cout << "  " << std::string(60, '-') << "\n";
        print_table_row("mean",   mean(json_rt),           mean(mp_rt),           mean(mp_inf),           mean(json_oh),           mean(mp_oh));
        print_table_row("min",    vmin(json_rt),           vmin(mp_rt),           vmin(mp_inf),           vmin(json_oh),           vmin(mp_oh));
        print_table_row("max",    vmax(json_rt),           vmax(mp_rt),           vmax(mp_inf),           vmax(json_oh),           vmax(mp_oh));
        print_table_row("p95",    percentile(json_rt, 95), percentile(mp_rt, 95), percentile(mp_inf, 95), percentile(json_oh, 95), percentile(mp_oh, 95));

        std::cout << "\n  Control loop budget (100 Hz): " << BUDGET_MS << " ms\n";
        printf("  JSON:    %s\n", mean(json_rt) < BUDGET_MS ? "PASS" : "WARN — exceeds budget");
        printf("  msgpack: %s\n", mean(mp_rt)   < BUDGET_MS ? "PASS" : "WARN — exceeds budget");
    }

    std::cout << "\n" << std::string(60, '=') << "\n";

    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return 0;
}
