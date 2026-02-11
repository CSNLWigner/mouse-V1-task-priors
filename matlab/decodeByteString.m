function array = decodeByteString(byteString)
	byteStream = uint8(hex2dec(reshape(byteString(3:end), 2, [])'));
	array = getArrayFromByteStream(byteStream);
end
