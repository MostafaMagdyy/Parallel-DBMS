#include <iostream>
#include <chrono>
#include <cstring>
#include <string>
#include <cstdint>
#include <bitset>
using namespace std;
#define MAX_STRING_LENGTH 30
int main() {
    string f="MOSTAFA";
    string s="mostafa";
string string_data;
string_data.reserve(2 * MAX_STRING_LENGTH);
auto slot_off = 2 * MAX_STRING_LENGTH;
string_data.resize(string_data.size() + MAX_STRING_LENGTH, '\0');
auto copy_len = min(value.size(), size_t(MAX_STRING_LENGTH - 1));
std::memcpy(&string_data[slot_off], value.data(), copy_len);
}