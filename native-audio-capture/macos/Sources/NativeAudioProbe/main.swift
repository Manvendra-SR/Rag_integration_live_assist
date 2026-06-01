import AVFoundation
import AudioToolbox
import CoreAudio
import CoreMedia
import Foundation
import ScreenCaptureKit

struct Level {
    let rms: Float
    let peak: Float
}

struct SampleLevelResult {
    let level: Level
    let byteCount: Int
    let parsedFormat: String
}

struct ProbeOptions {
    let includeOwnAudio: Bool
    let debugFormat: Bool
    let captureMode: String

    static func parse(_ arguments: [String]) -> ProbeOptions {
        var includeOwnAudio = false
        var debugFormat = false
        var captureMode = "display"

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "--include-own-audio" {
                includeOwnAudio = true
            } else if argument == "--debug-format" {
                debugFormat = true
            } else if argument == "--capture-mode", index + 1 < arguments.count {
                captureMode = arguments[index + 1]
                index += 1
            } else if argument.hasPrefix("--capture-mode=") {
                captureMode = String(argument.dropFirst("--capture-mode=".count))
            } else if argument == "--help" || argument == "-h" {
                log("""
                Usage:
                  native-audio-probe [--debug-format] [--include-own-audio] [--capture-mode display]

                Options:
                  --debug-format        Print ScreenCaptureKit audio format metadata.
                  --include-own-audio   Set excludesCurrentProcessAudio=false.
                  --capture-mode        Currently supports display. Other values log and fall back.
                """)
            }
            index += 1
        }

        return ProbeOptions(
            includeOwnAudio: includeOwnAudio,
            debugFormat: debugFormat,
            captureMode: captureMode
        )
    }
}

func log(_ message: String) {
    let line = "\(message)\n"
    if let data = line.data(using: .utf8) {
        FileHandle.standardOutput.write(data)
    }
}

func waitForever() async {
    while true {
        try? await Task.sleep(nanoseconds: 1_000_000_000)
    }
}

func levelFromFloatBuffer(_ buffer: AVAudioPCMBuffer) -> Level {
    guard let channelData = buffer.floatChannelData else {
        return Level(rms: 0, peak: 0)
    }

    let channels = Int(buffer.format.channelCount)
    let frames = Int(buffer.frameLength)
    if channels == 0 || frames == 0 {
        return Level(rms: 0, peak: 0)
    }

    var sumSquares: Float = 0
    var peak: Float = 0
    var sampleCount = 0

    for channel in 0..<channels {
        let samples = channelData[channel]
        for index in 0..<frames {
            let value = samples[index]
            sumSquares += value * value
            peak = max(peak, abs(value))
            sampleCount += 1
        }
    }

    if sampleCount == 0 {
        return Level(rms: 0, peak: 0)
    }

    return Level(rms: sqrt(sumSquares / Float(sampleCount)), peak: peak)
}

func fourCharacterCode(_ value: AudioFormatID) -> String {
    let bytes: [UInt8] = [
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff)
    ]

    let printable = bytes.allSatisfy { byte in
        byte >= 32 && byte <= 126
    }

    if printable, let string = String(bytes: bytes, encoding: .ascii) {
        return string
    }

    return "\(value)"
}

func audioFormatMetadata(from sampleBuffer: CMSampleBuffer, byteCount: Int, parsedFormat: String) -> String {
    let sampleCount = CMSampleBufferGetNumSamples(sampleBuffer)
    let dataReady = CMSampleBufferDataIsReady(sampleBuffer)

    guard
        let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
        let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)
    else {
        return "sample_count=\(sampleCount) data_ready=\(dataReady) byte_count=\(byteCount) parsed_format=\(parsedFormat) format=missing"
    }

    let audioFormat = streamDescription.pointee
    return [
        "sample_count=\(sampleCount)",
        "channel_count=\(audioFormat.mChannelsPerFrame)",
        "sample_rate=\(String(format: "%.1f", audioFormat.mSampleRate))",
        "format_id=\(fourCharacterCode(audioFormat.mFormatID))",
        "format_flags=\(audioFormat.mFormatFlags)",
        "bits_per_channel=\(audioFormat.mBitsPerChannel)",
        "bytes_per_frame=\(audioFormat.mBytesPerFrame)",
        "data_ready=\(dataReady)",
        "byte_count=\(byteCount)",
        "parsed_format=\(parsedFormat)"
    ].joined(separator: " ")
}

func levelFromSampleBuffer(_ sampleBuffer: CMSampleBuffer) -> SampleLevelResult {
    var blockBuffer: CMBlockBuffer?

    var bufferListSizeNeeded = 0
    let sizeStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: &bufferListSizeNeeded,
        bufferListOut: nil,
        bufferListSize: 0,
        blockBufferAllocator: nil,
        blockBufferMemoryAllocator: nil,
        flags: 0,
        blockBufferOut: nil
    )

    guard sizeStatus == noErr, bufferListSizeNeeded > 0 else {
        return SampleLevelResult(
            level: Level(rms: 0, peak: 0),
            byteCount: 0,
            parsedFormat: "buffer_list_unavailable"
        )
    }

    let rawAudioBufferList = UnsafeMutableRawPointer.allocate(
        byteCount: bufferListSizeNeeded,
        alignment: MemoryLayout<AudioBufferList>.alignment
    )
    defer {
        rawAudioBufferList.deallocate()
    }

    let audioBufferList = rawAudioBufferList.bindMemory(to: AudioBufferList.self, capacity: 1)

    let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: nil,
        bufferListOut: audioBufferList,
        bufferListSize: bufferListSizeNeeded,
        blockBufferAllocator: nil,
        blockBufferMemoryAllocator: nil,
        flags: 0,
        blockBufferOut: &blockBuffer
    )

    guard status == noErr else {
        return SampleLevelResult(
            level: Level(rms: 0, peak: 0),
            byteCount: 0,
            parsedFormat: "buffer_list_error_\(status)"
        )
    }

    let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer)
    guard
        let streamDescription = formatDescription.flatMap({
            CMAudioFormatDescriptionGetStreamBasicDescription($0)
        })
    else {
        let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
        let byteCount = buffers.reduce(0) { total, buffer in
            total + Int(buffer.mDataByteSize)
        }
        return SampleLevelResult(
            level: Level(rms: 0, peak: 0),
            byteCount: byteCount,
            parsedFormat: "format_description_missing"
        )
    }

    let audioFormat = streamDescription.pointee
    let bytesPerSample = Int(audioFormat.mBitsPerChannel / 8)
    let isFloat = (audioFormat.mFormatFlags & kAudioFormatFlagIsFloat) != 0
    let isSignedInteger = (audioFormat.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0

    var sumSquares: Float = 0
    var peak: Float = 0
    var sampleCount = 0
    var byteCountTotal = 0
    var parsedFormat = "unsupported"

    let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
    for buffer in buffers {
        byteCountTotal += Int(buffer.mDataByteSize)
        guard let data = buffer.mData else {
            continue
        }

        let byteCount = Int(buffer.mDataByteSize)
        if isFloat && bytesPerSample == 4 {
            parsedFormat = "float32"
            let samples = data.bindMemory(to: Float.self, capacity: byteCount / 4)
            for index in 0..<(byteCount / 4) {
                let value = samples[index]
                sumSquares += value * value
                peak = max(peak, abs(value))
                sampleCount += 1
            }
        } else if isSignedInteger && bytesPerSample == 2 {
            parsedFormat = "int16"
            let samples = data.bindMemory(to: Int16.self, capacity: byteCount / 2)
            for index in 0..<(byteCount / 2) {
                let value = Float(samples[index]) / Float(Int16.max)
                sumSquares += value * value
                peak = max(peak, abs(value))
                sampleCount += 1
            }
        } else if isSignedInteger && bytesPerSample == 4 {
            parsedFormat = "int32"
            let samples = data.bindMemory(to: Int32.self, capacity: byteCount / 4)
            for index in 0..<(byteCount / 4) {
                let value = Float(samples[index]) / Float(Int32.max)
                sumSquares += value * value
                peak = max(peak, abs(value))
                sampleCount += 1
            }
        }
    }

    if sampleCount == 0 {
        return SampleLevelResult(
            level: Level(rms: 0, peak: 0),
            byteCount: byteCountTotal,
            parsedFormat: parsedFormat
        )
    }

    return SampleLevelResult(
        level: Level(rms: sqrt(sumSquares / Float(sampleCount)), peak: peak),
        byteCount: byteCountTotal,
        parsedFormat: parsedFormat
    )
}

final class MicProbe {
    private let engine = AVAudioEngine()

    func start() throws {
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            let level = levelFromFloatBuffer(buffer)
            log(String(format: "[Native Mic] rms=%.5f peak=%.5f", level.rms, level.peak))
        }

        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

@available(macOS 13.0, *)
final class SystemAudioProbe: NSObject, SCStreamOutput, SCStreamDelegate {
    private let options: ProbeOptions
    private var stream: SCStream?
    private var audioSampleIndex = 0

    init(options: ProbeOptions) {
        self.options = options
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        )

        log("[Native Shareable Content] displays=\(content.displays.count) apps=\(content.applications.count) windows=\(content.windows.count)")

        guard let display = content.displays.first else {
            throw NSError(
                domain: "NativeAudioProbe",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No display available for ScreenCaptureKit."]
            )
        }

        log("[Native Display] selected_id=\(display.displayID) width=\(display.width) height=\(display.height)")

        if options.captureMode != "display" {
            log("[Native Capture Mode] requested=\(options.captureMode) supported=false fallback=display")
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = !options.includeOwnAudio
        config.sampleRate = 48_000
        config.channelCount = 2
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        log("[Native System Config] capture_mode=display captures_audio=true excludes_current_process_audio=\(config.excludesCurrentProcessAudio) sample_rate=\(config.sampleRate) channel_count=\(config.channelCount)")

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: .global(qos: .userInitiated))
        try await stream.startCapture()
        self.stream = stream
        log("[Native System] capture_started=true")
    }

    func stop() async {
        guard let stream else {
            return
        }
        try? await stream.stopCapture()
        self.stream = nil
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio else {
            return
        }
        audioSampleIndex += 1
        let result = levelFromSampleBuffer(sampleBuffer)
        if options.debugFormat && (audioSampleIndex <= 5 || audioSampleIndex % 50 == 0) {
            log("[Native System Format] index=\(audioSampleIndex) \(audioFormatMetadata(from: sampleBuffer, byteCount: result.byteCount, parsedFormat: result.parsedFormat))")
        }
        log(String(
            format: "[Native System] index=%d rms=%.5f peak=%.5f byte_count=%d parsed_format=%@",
            audioSampleIndex,
            result.level.rms,
            result.level.peak,
            result.byteCount,
            result.parsedFormat
        ))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("[Native Error] source=system type=\(type(of: error)) message=\"\(error.localizedDescription)\"")
    }
}

func defaultDeviceName(selector: AudioObjectPropertySelector, scope: AudioObjectPropertyScope) -> String {
    var deviceID = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )

    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &address,
        0,
        nil,
        &size,
        &deviceID
    )

    guard status == noErr else {
        return "unknown"
    }

    var name: CFString = "" as CFString
    size = UInt32(MemoryLayout<CFString>.size)
    var nameAddress = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )

    let nameStatus = AudioObjectGetPropertyData(
        deviceID,
        &nameAddress,
        0,
        nil,
        &size,
        &name
    )

    guard nameStatus == noErr else {
        return "unknown"
    }

    return name as String
}

@main
struct NativeAudioProbe {
    static func main() async {
        let options = ProbeOptions.parse(CommandLine.arguments)
        let osVersion = ProcessInfo.processInfo.operatingSystemVersion
        log("[Native OS] macOS=\(osVersion.majorVersion).\(osVersion.minorVersion).\(osVersion.patchVersion)")
        log("[Native Options] debug_format=\(options.debugFormat) include_own_audio=\(options.includeOwnAudio) capture_mode=\(options.captureMode)")
        log("[Native Permissions] required=\"Microphone plus Screen & System Audio Recording for Terminal/iTerm/VS Code or the packaged helper; quit and reopen the host app after changing permissions.\"")

        let inputName = defaultDeviceName(
            selector: kAudioHardwarePropertyDefaultInputDevice,
            scope: kAudioDevicePropertyScopeInput
        )
        let outputName = defaultDeviceName(
            selector: kAudioHardwarePropertyDefaultOutputDevice,
            scope: kAudioDevicePropertyScopeOutput
        )
        log("[Native Devices] default_input=\"\(inputName)\" default_output=\"\(outputName)\"")

        let micProbe = MicProbe()
        do {
            try micProbe.start()
            log("[Native Mic] capture_started=true")
        } catch {
            log("[Native Error] source=mic type=\(type(of: error)) message=\"\(error.localizedDescription)\"")
        }

        if #available(macOS 13.0, *) {
            let systemProbe = SystemAudioProbe(options: options)
            do {
                try await systemProbe.start()
                log("[Native Probe] running=true press_ctrl_c_to_stop=true")
                await waitForever()
            } catch {
                log("[Native Error] source=system type=\(type(of: error)) message=\"\(error.localizedDescription)\"")
                log("[Native Probe] mic_only_running=true press_ctrl_c_to_stop=true")
                await waitForever()
            }
        } else {
            log("[Native Error] source=system type=UnsupportedOS message=\"ScreenCaptureKit audio requires macOS 13+.\"")
            log("[Native Probe] mic_only_running=true press_ctrl_c_to_stop=true")
            await waitForever()
        }
    }
}
