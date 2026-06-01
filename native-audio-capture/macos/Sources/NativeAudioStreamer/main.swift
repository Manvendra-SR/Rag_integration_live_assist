import AVFoundation
import AudioToolbox
import CoreAudio
import CoreMedia
import Foundation
import ScreenCaptureKit

struct StreamerOptions {
    let wsURL: URL
    let callID: String
    let sessionID: String
    let includeOwnAudio: Bool
    let chunkFrames: Int
    let micOnlyDiagnostic: Bool
    let diagnostics: Bool

    static func parse(_ arguments: [String]) -> StreamerOptions {
        var wsURL = URL(string: "ws://127.0.0.1:8089")!
        var callID = "desktop-call-\(UUID().uuidString)"
        var sessionID = "desktop-session-\(UUID().uuidString)"
        var includeOwnAudio = true
        var chunkFrames = 4096
        var micOnlyDiagnostic = false
        var diagnostics = false

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "--ws-url", index + 1 < arguments.count {
                wsURL = URL(string: arguments[index + 1]) ?? wsURL
                index += 1
            } else if argument.hasPrefix("--ws-url=") {
                wsURL = URL(string: String(argument.dropFirst("--ws-url=".count))) ?? wsURL
            } else if argument == "--call-id", index + 1 < arguments.count {
                callID = arguments[index + 1]
                index += 1
            } else if argument.hasPrefix("--call-id=") {
                callID = String(argument.dropFirst("--call-id=".count))
            } else if argument == "--session-id", index + 1 < arguments.count {
                sessionID = arguments[index + 1]
                index += 1
            } else if argument.hasPrefix("--session-id=") {
                sessionID = String(argument.dropFirst("--session-id=".count))
            } else if argument == "--exclude-own-audio" {
                includeOwnAudio = false
            } else if argument == "--chunk-frames", index + 1 < arguments.count {
                chunkFrames = Int(arguments[index + 1]) ?? chunkFrames
                index += 1
            } else if argument.hasPrefix("--chunk-frames=") {
                chunkFrames = Int(String(argument.dropFirst("--chunk-frames=".count))) ?? chunkFrames
            } else if argument == "--mic-only-diagnostic" {
                micOnlyDiagnostic = true
            } else if argument == "--diagnostics" {
                diagnostics = true
            } else if argument == "--help" || argument == "-h" {
                log("""
                Usage:
                  native-audio-streamer --ws-url ws://127.0.0.1:8089 --call-id desktop-call-id --session-id desktop-session-id

                Options:
                  --exclude-own-audio  Set ScreenCaptureKit excludesCurrentProcessAudio=true.
                  --chunk-frames       PCM frames per websocket send. Default: 4096.
                  --mic-only-diagnostic Start mic capture only and pad system channel with silence.
                  --diagnostics         Print packet-level diagnostics and send chunk metadata.
                """)
            }
            index += 1
        }

        return StreamerOptions(
            wsURL: wsURL,
            callID: callID,
            sessionID: sessionID,
            includeOwnAudio: includeOwnAudio,
            chunkFrames: chunkFrames,
            micOnlyDiagnostic: micOnlyDiagnostic,
            diagnostics: diagnostics
        )
    }

    var websocketURL: URL {
        var components = URLComponents(url: wsURL, resolvingAgainstBaseURL: false)!
        var queryItems = components.queryItems ?? []
        queryItems.append(URLQueryItem(name: "call_id", value: callID))
        queryItems.append(URLQueryItem(name: "session_id", value: sessionID))
        components.queryItems = queryItems
        return components.url ?? wsURL
    }
}

struct Level {
    let rms: Float
    let peak: Float
}

struct DeviceInfo {
    let name: String
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

func level(_ samples: [Float]) -> Level {
    if samples.isEmpty {
        return Level(rms: 0, peak: 0)
    }

    var sumSquares: Float = 0
    var peak: Float = 0
    for sample in samples {
        sumSquares += sample * sample
        peak = max(peak, abs(sample))
    }

    return Level(rms: sqrt(sumSquares / Float(samples.count)), peak: peak)
}

func audioFormatDescription(_ format: AVAudioFormat) -> String {
    return "sample_rate=\(format.sampleRate) channels=\(format.channelCount) common_format=\(format.commonFormat.rawValue) interleaved=\(format.isInterleaved)"
}

func monoSamples(from buffer: AVAudioPCMBuffer, source: String = "mic") -> [Float] {
    let channels = Int(buffer.format.channelCount)
    let frames = Int(buffer.frameLength)
    if channels == 0 || frames == 0 {
        return []
    }

    var output = Array(repeating: Float(0), count: frames)

    if let channelData = buffer.floatChannelData {
        for channel in 0..<channels {
            let samples = channelData[channel]
            for index in 0..<frames {
                output[index] += samples[index] / Float(channels)
            }
        }
        return output
    }

    if let channelData = buffer.int16ChannelData {
        for channel in 0..<channels {
            let samples = channelData[channel]
            for index in 0..<frames {
                output[index] += (Float(samples[index]) / Float(Int16.max)) / Float(channels)
            }
        }
        return output
    }

    if let channelData = buffer.int32ChannelData {
        for channel in 0..<channels {
            let samples = channelData[channel]
            for index in 0..<frames {
                output[index] += (Float(samples[index]) / Float(Int32.max)) / Float(channels)
            }
        }
        return output
    }

    log("[Native Sample Warning] source=\(source) reason=unsupported_pcm_format \(audioFormatDescription(buffer.format))")
    return []
}

func monoSamples(from sampleBuffer: CMSampleBuffer) -> ([Float], Double) {
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
        return ([], 48_000)
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
        return ([], 48_000)
    }

    guard
        let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
        let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)
    else {
        return ([], 48_000)
    }

    let audioFormat = streamDescription.pointee
    let sampleRate = audioFormat.mSampleRate
    let bytesPerSample = Int(audioFormat.mBitsPerChannel / 8)
    let isFloat = (audioFormat.mFormatFlags & kAudioFormatFlagIsFloat) != 0
    let isSignedInteger = (audioFormat.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0
    let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)

    var channels: [[Float]] = []
    for buffer in buffers {
        guard let data = buffer.mData else {
            continue
        }

        let byteCount = Int(buffer.mDataByteSize)
        var channelSamples: [Float] = []
        if isFloat && bytesPerSample == 4 {
            let samples = data.bindMemory(to: Float.self, capacity: byteCount / 4)
            channelSamples.reserveCapacity(byteCount / 4)
            for index in 0..<(byteCount / 4) {
                channelSamples.append(samples[index])
            }
        } else if isSignedInteger && bytesPerSample == 2 {
            let samples = data.bindMemory(to: Int16.self, capacity: byteCount / 2)
            channelSamples.reserveCapacity(byteCount / 2)
            for index in 0..<(byteCount / 2) {
                channelSamples.append(Float(samples[index]) / Float(Int16.max))
            }
        } else if isSignedInteger && bytesPerSample == 4 {
            let samples = data.bindMemory(to: Int32.self, capacity: byteCount / 4)
            channelSamples.reserveCapacity(byteCount / 4)
            for index in 0..<(byteCount / 4) {
                channelSamples.append(Float(samples[index]) / Float(Int32.max))
            }
        }

        if !channelSamples.isEmpty {
            channels.append(channelSamples)
        }
    }

    guard let first = channels.first else {
        return ([], sampleRate)
    }

    if channels.count == 1 {
        return (first, sampleRate)
    }

    let frameCount = channels.map(\.count).min() ?? 0
    var output = Array(repeating: Float(0), count: frameCount)
    for channel in channels {
        for index in 0..<frameCount {
            output[index] += channel[index] / Float(channels.count)
        }
    }

    return (output, sampleRate)
}

func resampleLinear(_ input: [Float], from inputRate: Double, to outputRate: Double = 16_000) -> [Float] {
    if input.isEmpty {
        return []
    }
    if abs(inputRate - outputRate) < 1 {
        return input
    }

    let ratio = inputRate / outputRate
    let outputCount = Int(Double(input.count) / ratio)
    if outputCount <= 0 {
        return []
    }

    var output: [Float] = []
    output.reserveCapacity(outputCount)
    for outputIndex in 0..<outputCount {
        let sourcePosition = Double(outputIndex) * ratio
        let lowerIndex = Int(sourcePosition)
        let upperIndex = min(lowerIndex + 1, input.count - 1)
        let fraction = Float(sourcePosition - Double(lowerIndex))
        let lower = input[lowerIndex]
        let upper = input[upperIndex]
        output.append(lower + (upper - lower) * fraction)
    }
    return output
}

func appendPCM16Stereo(left: [Float], right: [Float], to data: inout Data) {
    let frameCount = min(left.count, right.count)
    for index in 0..<frameCount {
        let leftSample = max(Float(-1), min(Float(1), left[index]))
        let rightSample = max(Float(-1), min(Float(1), right[index]))
        var leftInt = Int16(leftSample < 0 ? leftSample * 32768 : leftSample * 32767)
        var rightInt = Int16(rightSample < 0 ? rightSample * 32768 : rightSample * 32767)
        data.append(Data(bytes: &leftInt, count: MemoryLayout<Int16>.size))
        data.append(Data(bytes: &rightInt, count: MemoryLayout<Int16>.size))
    }
}

final class WebSocketClient {
    private let url: URL
    private let diagnostics: Bool
    private let session: URLSession
    private let sendQueue = DispatchQueue(label: "native-audio-streamer.websocket-send")
    private let stateQueue = DispatchQueue(label: "native-audio-streamer.websocket-state")
    private var task: URLSessionWebSocketTask?
    private var connected = false
    private var reconnecting = false

    init(url: URL, diagnostics: Bool) {
        self.url = url
        self.diagnostics = diagnostics
        session = URLSession(configuration: .default)
    }

    func connect() {
        stateQueue.async {
            self.connectLocked()
        }
    }

    private func connectLocked() {
        task?.cancel(with: .goingAway, reason: nil)
        let newTask = session.webSocketTask(with: url)
        task = newTask
        connected = true
        reconnecting = false
        newTask.resume()
        log("[Native Stream] connected=true target=\"\(url.absoluteString)\"")
        receiveLoop(task: newTask)
    }

    func send(_ data: Data, chunkIndex: Int, systemRms: Float = 0, micRms: Float = 0) {
        sendQueue.async { [weak self] in
            guard let self else {
                return
            }
            guard let task = self.stateQueue.sync(execute: { self.task }) else {
                log("[Native WS Drop] chunk=\(chunkIndex) reason=no_task")
                self.scheduleReconnect(reason: "send_no_task")
                return
            }

            let sentAtMs = Int(Date().timeIntervalSince1970 * 1000)
            let meta: [String: Any] = [
                "type": "AUDIO_CHUNK_META",
                "chunk_index": chunkIndex,
                "bytes": data.count,
                "sent_at_ms": sentAtMs,
                "system_rms": systemRms,
                "mic_rms": micRms
            ]
            let metaData = try? JSONSerialization.data(withJSONObject: meta, options: [])
            let metaText = metaData.flatMap { String(data: $0, encoding: .utf8) } ?? ""

            let sendAudio = {
                task.send(.data(data)) { [weak self] error in
                    if let error {
                        log("[Native WS Error] chunk=\(chunkIndex) message=\"\(error.localizedDescription)\"")
                        self?.scheduleReconnect(reason: "send_error")
                        return
                    }
                    if self?.diagnostics == true {
                        log("[Native WS] sent_chunk=\(chunkIndex) bytes=\(data.count)")
                    }
                }
            }

            guard self.diagnostics else {
                sendAudio()
                return
            }

            guard !metaText.isEmpty else {
                log("[Native WS Meta Error] chunk=\(chunkIndex) reason=json_encode_failed")
                sendAudio()
                return
            }

            task.send(.string(metaText)) { [weak self] error in
                if let error {
                    log("[Native WS Meta Error] chunk=\(chunkIndex) message=\"\(error.localizedDescription)\"")
                    self?.scheduleReconnect(reason: "send_error")
                    return
                }
                sendAudio()
            }
        }
    }

    func close() {
        stateQueue.async {
            self.connected = false
            self.reconnecting = false
            self.task?.cancel(with: .goingAway, reason: nil)
            self.task = nil
        }
    }

    private func receiveLoop(task currentTask: URLSessionWebSocketTask) {
        currentTask.receive { [weak self] result in
            guard let self, self.connected else {
                return
            }

            let isCurrentTask = self.stateQueue.sync { self.task === currentTask }
            guard isCurrentTask else {
                return
            }

            switch result {
            case .success(.string(let text)):
                log("[Native Backend Event] \(text)")
            case .success(.data(let data)):
                if let text = String(data: data, encoding: .utf8) {
                    log("[Native Backend Event] \(text)")
                }
            case .failure(let error):
                log("[Native WS Receive Error] message=\"\(error.localizedDescription)\"")
                self.scheduleReconnect(reason: "receive_error")
                return
            @unknown default:
                log("[Native WS Receive Error] message=\"unknown websocket message\"")
                self.scheduleReconnect(reason: "unknown_receive")
                return
            }

            self.receiveLoop(task: currentTask)
        }
    }

    private func scheduleReconnect(reason: String) {
        stateQueue.async {
            guard self.connected, !self.reconnecting else {
                return
            }
            self.reconnecting = true
            self.task?.cancel(with: .goingAway, reason: nil)
            self.task = nil
            log("[Native WS Reconnect] reason=\(reason) retry_seconds=2")
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2) {
                self.stateQueue.async {
                    guard self.connected else {
                        return
                    }
                    self.connectLocked()
                }
            }
        }
    }
}

final class StereoMixer {
    private let queue = DispatchQueue(label: "native-audio-streamer.stereo-mixer")
    private let chunkFrames: Int
    private let sender: WebSocketClient
    private let diagnostics: Bool
    private var leftBuffer: [Float] = []
    private var rightBuffer: [Float] = []
    private var sentChunks = 0
    private var lastEnergyLogAt = Date.distantPast
    private var lastDominanceLogAt = Date.distantPast

    init(chunkFrames: Int, sender: WebSocketClient, diagnostics: Bool) {
        self.chunkFrames = chunkFrames
        self.sender = sender
        self.diagnostics = diagnostics
        leftBuffer.reserveCapacity(chunkFrames * 2)
        rightBuffer.reserveCapacity(chunkFrames * 2)
    }

    func appendSystem(_ samples: [Float], sampleRate: Double) {
        let resampled = resampleLinear(samples, from: sampleRate)
        queue.async {
            self.leftBuffer.append(contentsOf: resampled)
            self.flushReadyChunks()
        }
    }

    func appendMic(_ samples: [Float], sampleRate: Double) {
        let resampled = resampleLinear(samples, from: sampleRate)
        queue.async {
            self.rightBuffer.append(contentsOf: resampled)
            self.flushReadyChunks()
        }
    }

    private func flushReadyChunks() {
        let now = Date()
        if now.timeIntervalSince(lastEnergyLogAt) >= 1.0 {
            let leftWindow = Array(leftBuffer.prefix(min(leftBuffer.count, chunkFrames)))
            let rightWindow = Array(rightBuffer.prefix(min(rightBuffer.count, chunkFrames)))
            let leftLevel = level(leftWindow)
            let rightLevel = level(rightWindow)
            log(String(
                format: "[Native Energy] system_rms=%.5f system_peak=%.5f mic_rms=%.5f mic_peak=%.5f left_buffer=%d right_buffer=%d",
                leftLevel.rms,
                leftLevel.peak,
                rightLevel.rms,
                rightLevel.peak,
                leftBuffer.count,
                rightBuffer.count
            ))
            lastEnergyLogAt = now
        }

        while leftBuffer.count >= chunkFrames || rightBuffer.count >= chunkFrames {
            let left = takeFrames(from: &leftBuffer, count: chunkFrames)
            let right = takeFrames(from: &rightBuffer, count: chunkFrames)
            if diagnostics {
                logDominance(left: left, right: right)
            }
            let leftLevel = level(left)
            let rightLevel = level(right)
            var payload = Data(capacity: chunkFrames * 2 * MemoryLayout<Int16>.size)
            appendPCM16Stereo(left: left, right: right, to: &payload)
            sentChunks += 1
            sender.send(payload, chunkIndex: sentChunks, systemRms: leftLevel.rms, micRms: rightLevel.rms)
        }
    }

    private func logDominance(left: [Float], right: [Float]) {
        let now = Date()
        guard now.timeIntervalSince(lastDominanceLogAt) >= 1.0 else {
            return
        }

        let leftLevel = level(left)
        let rightLevel = level(right)
        let dominant: String
        if leftLevel.rms > rightLevel.rms * 2.5 {
            dominant = "system"
        } else if rightLevel.rms > leftLevel.rms * 2.5 {
            dominant = "mic"
        } else {
            dominant = "mixed"
        }

        log(String(
            format: "[Native Dominance] dominant=%@ system_rms=%.5f mic_rms=%.5f",
            dominant,
            leftLevel.rms,
            rightLevel.rms
        ))
        lastDominanceLogAt = now
    }

    private func takeFrames(from buffer: inout [Float], count: Int) -> [Float] {
        if buffer.count >= count {
            let frames = Array(buffer.prefix(count))
            buffer.removeFirst(count)
            return frames
        }

        let frames = buffer + Array(repeating: Float(0), count: count - buffer.count)
        buffer.removeAll(keepingCapacity: true)
        return frames
    }
}

final class MicCapture {
    private var engine = AVAudioEngine()
    private let mixer: StereoMixer
    private let diagnostics: Bool
    private var lastEnergyLogAt = Date.distantPast
    private var deviceListenerInstalled = false
    private var callbackSeq = 0
    private var timeoutRestartAttempted = false
    private var activeDeviceName = "unknown"
    private var tapInstalled = false

    init(mixer: StereoMixer, diagnostics: Bool) {
        self.mixer = mixer
        self.diagnostics = diagnostics
    }

    func start() throws {
        try startEngine()
        installDefaultInputListener()
    }

    private func startEngine() throws {
        if engine.isRunning {
            engine.stop()
        }
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        engine.reset()
        engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        let deviceName = defaultDeviceName(
            selector: kAudioHardwarePropertyDefaultInputDevice,
            scope: kAudioDevicePropertyScopeInput
        )
        activeDeviceName = deviceName
        callbackSeq = 0
        log("[Native Mic Format] device=\"\(deviceName)\" \(audioFormatDescription(format))")
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: nil) { [weak self] buffer, _ in
            guard let self else {
                return
            }
            self.callbackSeq += 1
            let samples = monoSamples(from: buffer, source: "mic")
            self.logMicCallback(samples: samples, frames: Int(buffer.frameLength), format: buffer.format)
            self.mixer.appendMic(samples, sampleRate: buffer.format.sampleRate)
        }
        tapInstalled = true
        try engine.start()
        log("[Native Mic] capture_started=true device=\"\(deviceName)\" sample_rate=\(format.sampleRate) channels=\(format.channelCount)")
        scheduleCallbackTimeoutCheck(deviceName: deviceName)
    }

    private func restart(reason: String) {
        guard diagnostics else {
            log("[Native Mic Warning] reason=\(reason) action=manual_restart_required message=\"Stop and Start Listening to switch or recover microphone input.\"")
            return
        }
        do {
            log("[Native Mic Restart] reason=\(reason)")
            try startEngine()
        } catch {
            log("[Native Mic Warning] source=mic_restart type=\(type(of: error)) message=\"\(error.localizedDescription)\" action=continue_without_restart")
        }
    }

    private func logMicCallback(samples: [Float], frames: Int, format: AVAudioFormat) {
        let now = Date()
        let micLevel = level(samples)
        if diagnostics && (callbackSeq <= 10 || callbackSeq % 25 == 0) {
            log(String(
                format: "[Native Mic Callback] seq=%d frames=%d samples=%d rms=%.5f peak=%.5f device=\"%@\"",
                callbackSeq,
                frames,
                samples.count,
                micLevel.rms,
                micLevel.peak,
                activeDeviceName
            ))
        }
        guard now.timeIntervalSince(lastEnergyLogAt) >= 1.0 else {
            return
        }
        log(String(format: "[Native Mic Energy] rms=%.5f peak=%.5f", micLevel.rms, micLevel.peak))
        lastEnergyLogAt = now
    }

    private func scheduleCallbackTimeoutCheck(deviceName: String) {
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 3) { [weak self] in
            guard let self else {
                return
            }
            if self.callbackSeq == 0 && !self.timeoutRestartAttempted {
                self.timeoutRestartAttempted = true
                if self.diagnostics {
                    log("[Native Mic Timeout] callbacks=0 default_input=\"\(deviceName)\" action=restart message=\"No microphone packets received. Check default input device and microphone permission.\"")
                    self.restart(reason: "mic_callback_timeout")
                } else {
                    log("[Native Mic Warning] callbacks=0 default_input=\"\(deviceName)\" action=manual_restart_required message=\"No microphone packets received. Check default input device and microphone permission, then Stop and Start Listening.\"")
                }
            }
        }
    }

    private func installDefaultInputListener() {
        guard !deviceListenerInstalled else {
            return
        }
        deviceListenerInstalled = true
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let selfPointer = Unmanaged.passUnretained(self).toOpaque()
        let status = AudioObjectAddPropertyListener(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            { _, _, _, clientData in
                guard let clientData else {
                    return noErr
                }
                let capture = Unmanaged<MicCapture>.fromOpaque(clientData).takeUnretainedValue()
                DispatchQueue.main.async {
                    let inputName = defaultDeviceName(
                        selector: kAudioHardwarePropertyDefaultInputDevice,
                        scope: kAudioDevicePropertyScopeInput
                    )
                    log("[Native Devices] default_input_changed=\"\(inputName)\"")
                    if capture.diagnostics {
                        capture.restart(reason: "default_input_changed")
                    } else {
                        log("[Native Mic Warning] reason=default_input_changed action=manual_restart_required message=\"Stop and Start Listening to switch microphone input.\"")
                    }
                }
                return noErr
            },
            selfPointer
        )
        if status != noErr {
            log("[Native Warning] source=mic_listener status=\(status)")
        }
    }
}

@available(macOS 13.0, *)
final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let includeOwnAudio: Bool
    private let mixer: StereoMixer
    private var stream: SCStream?

    init(includeOwnAudio: Bool, mixer: StereoMixer) {
        self.includeOwnAudio = includeOwnAudio
        self.mixer = mixer
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        )
        log("[Native Shareable Content] displays=\(content.displays.count) apps=\(content.applications.count) windows=\(content.windows.count)")

        guard let display = content.displays.first else {
            throw NSError(
                domain: "NativeAudioStreamer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No display available for ScreenCaptureKit."]
            )
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = !includeOwnAudio
        config.sampleRate = 48_000
        config.channelCount = 2
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: .global(qos: .userInitiated))
        try await stream.startCapture()
        self.stream = stream
        log("[Native System] capture_started=true excludes_current_process_audio=\(config.excludesCurrentProcessAudio)")
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio else {
            return
        }
        let (samples, sampleRate) = monoSamples(from: sampleBuffer)
        mixer.appendSystem(samples, sampleRate: sampleRate)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("[Native Error] source=system type=\(type(of: error)) message=\"\(error.localizedDescription)\"")
    }
}

func defaultDeviceID(selector: AudioObjectPropertySelector) -> AudioDeviceID {
    var deviceID = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
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
        return 0
    }
    return deviceID
}

func deviceName(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> String {
    guard deviceID != 0 else {
        return "unknown"
    }

    var name: CFString = "" as CFString
    var size = UInt32(MemoryLayout<CFString>.size)
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

func defaultDeviceName(selector: AudioObjectPropertySelector, scope: AudioObjectPropertyScope) -> String {
    return deviceName(defaultDeviceID(selector: selector), scope: scope)
}

func inputDevices() -> [DeviceInfo] {
    AVCaptureDevice.devices(for: .audio).map { DeviceInfo(name: $0.localizedName) }
}

@main
struct NativeAudioStreamer {
    static func main() async {
        let options = StreamerOptions.parse(CommandLine.arguments)
        let inputName = defaultDeviceName(
            selector: kAudioHardwarePropertyDefaultInputDevice,
            scope: kAudioDevicePropertyScopeInput
        )
        let outputName = defaultDeviceName(
            selector: kAudioHardwarePropertyDefaultOutputDevice,
            scope: kAudioDevicePropertyScopeOutput
        )
        log("[Native Devices] default_input=\"\(inputName)\" default_output=\"\(outputName)\"")
        if options.diagnostics {
            let inputDeviceList = inputDevices()
            let inputs = inputDeviceList
                .map { "name=\"\($0.name)\"" }
                .joined(separator: "; ")
            log("[Native Input Devices] count=\(inputDeviceList.count) devices=[\(inputs)]")
        }
        log("[Native Stream] target=\"\(options.websocketURL.absoluteString)\" chunk_frames=\(options.chunkFrames)")

        let webSocket = WebSocketClient(url: options.websocketURL, diagnostics: options.diagnostics)
        webSocket.connect()
        let mixer = StereoMixer(
            chunkFrames: options.chunkFrames,
            sender: webSocket,
            diagnostics: options.diagnostics
        )
        var micCapture: MicCapture?
        var systemCapture: AnyObject?

        do {
            micCapture = MicCapture(mixer: mixer, diagnostics: options.diagnostics)
            try micCapture?.start()
        } catch {
            log("[Native Error] source=mic type=\(type(of: error)) message=\"\(error.localizedDescription)\"")
        }

        if options.micOnlyDiagnostic {
            log("[Native System] skipped=true reason=mic_only_diagnostic")
        } else if #available(macOS 13.0, *) {
            do {
                let system = SystemAudioCapture(includeOwnAudio: options.includeOwnAudio, mixer: mixer)
                try await system.start()
                systemCapture = system
            } catch {
                log("[Native Error] source=system type=\(type(of: error)) message=\"\(error.localizedDescription)\"")
            }
        } else {
            log("[Native Error] source=system type=UnsupportedOS message=\"ScreenCaptureKit audio requires macOS 13+.\"")
        }

        _ = micCapture
        _ = systemCapture
        log("[Native Stream] running=true press_ctrl_c_to_stop=true")
        await waitForever()
    }
}
