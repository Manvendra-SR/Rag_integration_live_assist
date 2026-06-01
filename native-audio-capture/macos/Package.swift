// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "NativeAudioCapture",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "native-audio-probe", targets: ["NativeAudioProbe"]),
        .executable(name: "native-audio-streamer", targets: ["NativeAudioStreamer"])
    ],
    targets: [
        .executableTarget(
            name: "NativeAudioProbe",
            path: "Sources/NativeAudioProbe"
        ),
        .executableTarget(
            name: "NativeAudioStreamer",
            path: "Sources/NativeAudioStreamer"
        )
    ]
)
